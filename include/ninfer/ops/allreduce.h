#pragma once

// ninfer::ops - two-device collectives.
//
// These are the cross-device primitives every tensor-parallel (tp == 2) forward pass needs: one
// summing reduction for row-parallel projections, one row gather for the vocabulary-split output
// head. Both operate on an ExecutionContext holding exactly two DeviceContexts and use each
// device's own compute stream; no stream parameter is accepted, because "the stream a device
// executes on" is already a property of its DeviceContext and a second, disagreeing stream story
// would break the ordering guarantees documented below.
//
// TRANSPORT. Payloads move with cudaMemcpyAsync(cudaMemcpyDeviceToDevice) over unified virtual
// addresses, which every 64-bit Linux CUDA context has: a device pointer already names its
// device, so this one entry point expresses a cross-device transfer as well as a local one. When
// the driver grants peer access the copy is a direct device-to-device PCIe transfer; when it does
// not (GeForce-class boards refuse peer access), CUDA transparently stages the same copy through
// host memory. Both paths are correct and stream-ordered, so no caller and no test needs a
// peer-access branch; only bandwidth and latency differ. enable_peer_access() below reports which
// one is active.
//
// The equivalent cudaMemcpyPeerAsync entry point is deliberately NOT used: it is rejected inside a
// stream capture region (cudaErrorStreamCaptureUnsupported), which would make the whole
// tensor-parallel decode program uncapturable. See src/ops/common/allreduce.cu's pull_peer().
//
// CAPTURED MAILBOX. A PeerEvents instance may carry a PeerMailbox (PeerEvents::attach_mailbox,
// ops/peer_mailbox.h). An allreduce_sum call recorded while BOTH ranks' streams belong to one
// capture, whose payload is whole aligned 16-byte vectors within one mailbox slot, then becomes
// one exchange kernel per device that publishes through pinned host memory instead of the staged
// choreography below; its result is bit-identical. Every other call keeps the staged path.
//
// ORDERING. Every transfer is a PULL: rank r reads the peer's source into storage that rank r
// alone owns, on rank r's own stream. Nothing a rank owns is ever written by the peer's stream, so
// the classic push hazard -- the peer overwriting a staging buffer that this rank has not finished
// reading yet -- cannot exist. Each rank issues two event records and two event waits:
//
//   stream(r):  ...producer of the rank-r inputs...
//               record(inputs_ready[r])
//               wait(inputs_ready[1-r])              // peer's source is complete
//               cudaMemcpyAsync(peer source -> rank-r storage)
//               record(pull_done[r])
//               wait(pull_done[1-r])                 // peer has finished reading MY source
//               ...local combine, for allreduce_sum...
//
// The two waits carry the whole correctness argument:
//
//   * wait(inputs_ready[1-r]) makes the inbound copy read a completed peer source, both within a
//     call and across calls -- in call k+1 that event is recorded after call k's local work on the
//     peer's stream, so the next pull cannot start before the previous call finished with the
//     source it reads.
//   * wait(pull_done[1-r]) is the write-after-read barrier for THIS rank's own source: it
//     guarantees the peer has finished reading rank r's buffer before rank r (or the caller, on
//     return) overwrites it. For allreduce_sum that protects the in-place combine; for both ops it
//     is what makes back-to-back calls that reuse the same buffers safe.
//
// POST-CONDITION. On return each rank's stream is ordered after the peer's read of that rank's
// inputs. Therefore an unbounded sequence of calls sharing the same buffers, staging, and
// PeerEvents needs NO host synchronization between calls -- which is exactly what a captured
// CUDA graph replays, and what a decode token's collectives do. For the 27B model that is
// 128 all-reduces (64 layers x 2 row-parallel projections: the mixer output and the MLP down
// projection); the vocabulary-split logits are then pulled to rank 0 alone by the model's own
// one-sided gather, which follows the same record/wait discipline.
//
// Everything in a call is stream-ordered device work (memcpy, event record, event wait, kernel
// launch): no host round trip, no host-memory spin flag, no device or stream synchronization. That
// is what makes the sequence capturable. Measured: with both device streams enrolled in ONE
// capture -- the peer's stream joined to the origin's by an event fork -- the record/wait pairs
// above become graph EDGES rather than nodes, and the whole 128-collective decode program
// instantiates as a single cross-device cudaGraphExec. Event objects and staging storage are
// caller-owned and created once, because cudaEventCreate and cudaMalloc are not capturable.
//
// CALLER OBLIGATION -- the legacy default stream. These ops read their inputs on
// DeviceContext::stream, which core creates with cudaStreamNonBlocking and which therefore does
// NOT implicitly synchronize with a device's legacy default stream. Work issued with the plain
// cudaMemcpy/cudaMemset/<<<...>>> forms (no stream argument) lands on that default stream and is
// unordered against these collectives. A caller that stages inputs that way must retire them
// first (a device or stream synchronize, or an event); producing the inputs on
// DeviceContext::stream needs no extra step. This trap is invisible in single-device code, where
// the default stream orders everything.
//
// After a call returns the work is enqueued but not complete; the caller synchronizes each
// device's own stream, or lets subsequent stream-ordered work depend on it, as usual.

#include "core/device.h" // DeviceContext, ExecutionContext
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaEvent_t

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::ops {

class PeerMailbox; // see ops/peer_mailbox.h
class HostExchange;
class PeerLanes;

// Probes cudaDeviceCanAccessPeer in both directions and enables peer access on both devices only
// when both directions report support; a device that already had peer access enabled is left
// alone. Returns true when direct P2P is active for the pair, false when the driver denies it and
// the collectives below will therefore run over CUDA's host-staged transfer. Never fails on
// denial: the staged path is a supported transport, not an error.
//
// Call once during setup. cudaDeviceEnablePeerAccess is a context-level operation, not
// stream-ordered and not graph-capturable; it must never appear in a hot path.
bool enable_peer_access(const ExecutionContext& ec);

// The reusable cross-device ordering events: two per device, created on that device with timing
// disabled.
//
//   inputs_ready(r) - recorded on rank r's stream once the inputs rank r contributes are complete;
//                     the peer waits on it before reading them.
//   pull_done(r)    - recorded on rank r's stream once rank r's inbound copy has finished reading
//                     the peer's source; the peer waits on it before overwriting that source.
//
// Re-recording an event a previous call already recorded is well defined: cudaStreamWaitEvent
// captures the event's state at the time the wait is issued, and the collectives always issue a
// call's waits before the next call's records, so one instance serves an unbounded number of
// sequential calls. Instances are not thread-safe: one instance belongs to one stream pair. A
// moved-from instance holds no events and must not be passed to a collective (the ops reject it).
//
// An instance also names the stream pair's captured transport: null (the staged path for every
// call) until attach_mailbox() installs a PeerMailbox, which the caller keeps alive for as long as
// this instance is used and every graph captured through it exists.
class PeerEvents {
public:
    explicit PeerEvents(const ExecutionContext& ec);
    ~PeerEvents();

    PeerEvents(const PeerEvents&)            = delete;
    PeerEvents& operator=(const PeerEvents&) = delete;
    PeerEvents(PeerEvents&& other) noexcept;
    PeerEvents& operator=(PeerEvents&& other) noexcept;

    [[nodiscard]] cudaEvent_t inputs_ready(int rank) const noexcept {
        return inputs_ready_[static_cast<std::size_t>(rank)];
    }

    [[nodiscard]] cudaEvent_t pull_done(int rank) const noexcept {
        return pull_done_[static_cast<std::size_t>(rank)];
    }

    // False for a moved-from instance.
    [[nodiscard]] bool live() const noexcept {
        return inputs_ready_[0] != nullptr && inputs_ready_[1] != nullptr &&
               pull_done_[0] != nullptr && pull_done_[1] != nullptr;
    }

    // Selects `mailbox` (null: none) as the transport of this pair's captured all-reduces. It must
    // serve the ExecutionContext these events were created for. Captures already recorded keep
    // the transport they were recorded with.
    void attach_mailbox(PeerMailbox* mailbox) noexcept { mailbox_ = mailbox; }

    // Mutable through a const instance: a captured call claims the mailbox's next slot.
    [[nodiscard]] PeerMailbox* mailbox() const noexcept { return mailbox_; }

    // While a StagedScope of this instance is open, captured calls keep the staged path even with
    // a mailbox attached. A caller opens one around captured collectives that share a round with
    // cross-device copies: under WSL2 such a copy can stall behind a spinning exchange kernel on
    // its source device, and the pair then waits on each other until the spin limit (issue #1).
    // Like mailbox(), usable through a const instance: it only selects the transport of the
    // calls recorded inside the scope. Scopes nest.
    class StagedScope {
    public:
        explicit StagedScope(const PeerEvents& events) noexcept : events_(&events) {
            ++events_->staged_scopes_;
        }

        ~StagedScope() { --events_->staged_scopes_; }

        StagedScope(const StagedScope&)            = delete;
        StagedScope& operator=(const StagedScope&) = delete;

    private:
        const PeerEvents* events_;
    };

    [[nodiscard]] bool staged_only() const noexcept { return staged_scopes_ != 0; }

    // Selects `exchange` (null: none) as the transport of this pair's EAGER (uncaptured) staged
    // collectives: allreduce_sum() and allreduce_sum_f32() outside a capture move the peer's
    // operand through its pinned host buffers instead of the driver-staged cross-device copy.
    // Captured calls ignore it. It must serve the ExecutionContext these events were created for.
    void attach_host_exchange(HostExchange* exchange) noexcept { host_exchange_ = exchange; }

    [[nodiscard]] HostExchange* host_exchange() const noexcept { return host_exchange_; }

    // Selects `lanes` (null: none) as the transport of this pair's EAGER split-phase sums
    // (allreduce_sum_begin / allreduce_sum_finish), ahead of an attached HostExchange's lanes.
    // Every other collective ignores it. It must serve the ExecutionContext these events were
    // created for. The Program attaches it only with peer access and no HostExchange, or under
    // NINFER_TP_PEER_OVERLAP=force (a correctness test without P2P).
    void attach_peer_lanes(PeerLanes* lanes) noexcept { peer_lanes_ = lanes; }

    [[nodiscard]] PeerLanes* peer_lanes() const noexcept { return peer_lanes_; }

private:
    std::array<cudaEvent_t, 2> inputs_ready_{nullptr, nullptr};
    std::array<cudaEvent_t, 2> pull_done_{nullptr, nullptr};
    PeerMailbox* mailbox_        = nullptr;
    HostExchange* host_exchange_ = nullptr;
    PeerLanes* peer_lanes_       = nullptr;
    mutable int staged_scopes_   = 0;
};

/**
 * Eager transport of the staged collectives on boards without peer access (T56).
 *
 * Without P2P, cudaMemcpyAsync(DeviceToDevice) across the two devices is staged by the driver
 * through host memory and BLOCKS THE CALLING HOST THREAD until the copy is done (measured on CUDA
 * 13.1 / driver 580: 34-62 ms per call for a [2560,16384] FP32 partial), the two directions run
 * one after the other, and the host can enqueue nothing past the exchange. HostExchange moves the
 * same bytes asynchronously through two pinned host buffers of its own: rank r copies its
 * operand device-to-host on a side stream of its device in chunks (its own link, D2H direction)
 * and records one event per chunk; the peer's stream waits for each chunk and copies it
 * host-to-device into its staging (the peer's link, H2D direction). Both ranks run both legs at
 * once, the host never blocks, and the bits moved are the same as the driver-staged pull's.
 *
 * Ordering (the staged path's three phases, unchanged around it): the side stream starts after
 * inputs_ready(r); the peer's last host-to-device chunk precedes pull_done(1-r), which rank r's
 * stream waits for before its combine (so its operand is not overwritten while being read) and
 * which precedes the next call's inputs_ready(r) (so its host buffer is not overwritten while the
 * peer still reads it). The buffers grow on demand; growing synchronizes both devices first.
 */
class HostExchange {
public:
    explicit HostExchange(const ExecutionContext& ec, std::size_t chunk_bytes = 8u << 20);
    ~HostExchange();
    HostExchange(const HostExchange&)            = delete;
    HostExchange& operator=(const HostExchange&) = delete;

    // Phase B of a staged collective, eager: destination[r] (on dev r) receives source[1-r]'s
    // `bytes`. Enqueued on ec.dev[r]->stream after inputs_ready(r) of both ranks were recorded.
    void pull(const std::array<const void*, 2>& source, const std::array<void*, 2>& destination,
              std::size_t bytes, const ExecutionContext& ec, const PeerEvents& events);

    [[nodiscard]] bool serves(const ExecutionContext& ec) const noexcept;

    // T68: one-way eager copy of `bytes` from `source` (on dev `from`) to `destination` (on the
    // peer, dev 1 - from) through `from`'s pinned host buffer: device-to-host on `from`'s side
    // stream behind inputs_ready(from), then host-to-device on the peer's stream, chunk by chunk.
    // For memory the peer cannot address without P2P (the elastic VMM pool). Eager only: take the
    // exchange from eager_exchange(). Host buffer reuse follows pull(): the next exchange's
    // inputs_ready(from) must come after the peer's stream has passed this copy.
    void copy(int from, const void* source, void* destination, std::size_t bytes,
              const ExecutionContext& ec, const PeerEvents& events);

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    // ---- T62: the TP2 prefill MoE exchange, by column groups, with an optional narrow payload --
    //
    // One FP32 summing all-reduce of the per-rank partials P_r (contiguous FP32 [H, T], H = ne[0],
    // columns contiguous), issued in three steps instead of one call so that the exchange of the
    // first columns overlaps the computation of the later ones:
    //
    //   begin_sum_f32(...)            once, before any column of either rank is ready;
    //   columns_ready(r, first, n)    per rank, after the work that makes columns [first, first+n)
    //                                 of P_r final has been enqueued on ec.dev[r]->stream; the
    //                                 groups of a rank are consecutive and cover [0, T) exactly;
    //   finish_sum_f32()              once both ranks have reported all T columns.
    //
    // Result, both ranks, identical bits (the one FP32 addition is commutative):
    //
    //   FP32 payload:  S = fl32(P_0 + P_1)                         (== allreduce_sum_f32)
    //   FP16 payload:  S = fl32(f16(P_0) + f16(P_1))               f16 = RNE to binary16
    //   BF16 payload:  S = fl32(bf16(P_0) + bf16(P_1))             bf16 = RNE to bfloat16
    //   partial[r] = S;  bf16_out[r] = RNE_bf16(S) when bf16_out is given.
    //
    // Each rank rounds ITS OWN operand to the payload format too, so both ranks add the same two
    // values. A narrow payload halves the bytes on the links; f16 overflows to infinity above
    // 65504 (the caller guards the range; NINFER_TP_AR_MAXABS=1 reports per call the largest
    // |P_r| and the count of values beyond the FP16 range, synchronously: diagnostic only).
    //
    // Transport: rank r's group goes device-to-host on rank r's side stream (narrowed first on
    // rank r's stream into the first half of staging[r] for FP16/BF16) in chunks, each chunk
    // host-to-device on a receive stream of the peer into staging[1-r] (FP32) or into the second
    // half of staging[1-r] (FP16/BF16). finish waits on rank r's stream for both receive streams
    // (its inbound data, and the peer's reads of rank r's host buffer, which follow rank r's own
    // device-to-host copies) and combines. The receive stream of rank r starts each call behind
    // rank r's stream at begin (the previous call's combine has read staging[r]).
    // Eager only: begin returns false, enqueuing nothing, when either rank's stream is capturing.
    // `staging[r]`: FP32 of the partial's shape, disjoint from it. Not thread-safe; one call at a
    // time; the pinned buffers are shared with pull() (calls on the two paths are stream-ordered
    // through each rank's stream).
    //
    // INT8 (NINFER_TP_AR_PAYLOAD=int8, opt-in): per column `rows` int8 codes and rows / 32 FP16
    // scales, rows * 17 / 16 bytes (2720 instead of 5120 for 2560 rows). Per 32-row block s =
    // binary16 rounded up of min(max(amax / 127, 2^-14), 65504), q = rint(x / s) in [-127, 127],
    // value q * s: values beyond 127 * 65504 saturate, an all-zero block is 0, a block holding NaN
    // or Inf becomes NaN in all 32 rows. Lossy (on real Qwen3.8 Flash partials ~3.4-3.8x the BF16
    // payload's error); rows not a multiple of 32 use FP16.
    enum class Payload : int { FP32 = 0, FP16 = 1, BF16 = 2, INT8 = 3 };
    [[nodiscard]] bool begin_sum_f32(const std::array<Tensor, 2>& partial,
                                     const std::array<Tensor, 2>& staging,
                                     const std::array<Tensor, 2>* bf16_out, Payload payload,
                                     const ExecutionContext& ec);
    void columns_ready(int rank, std::int64_t first, std::int64_t count);
    void finish_sum_f32();
    // Closes the open call without combining, for a caller that failed between begin_sum_f32 and
    // finish_sum_f32: each rank's stream waits for the copies already enqueued, after which
    // partial, staging and the pinned buffers may be reused and the next begin_sum_f32 opens.
    // partial[r] and bf16_out[r] are left unspecified. Does nothing without an open call; CUDA
    // errors are ignored (the caller is already propagating a failure).
    void abandon_sum_f32() noexcept;

    // Columns of rank r reported in the open call (0 when none is open).
    [[nodiscard]] std::int64_t columns_done(int rank) const noexcept {
        return pending_.active ? pending_.done[static_cast<std::size_t>(rank)] : 0;
    }

    // ---- Split-phase BF16 summing all-reduce on independent lanes (27B prefill overlap) ------
    //
    // allreduce_sum() of `buffer` through `staging` (same contract, same bits: the local combine
    // is the same residual_add body), cut in two so that other work enqueued on the ranks'
    // streams between the halves runs while the bytes move:
    //
    //   sum_begin(lane, ...)   both operands buffer[r] are complete on ec.dev[r]->stream: rank r's
    //                          operand goes device-to-host into the lane's own pinned buffer on a
    //                          send stream of rank r, in chunks, and each chunk host-to-device into
    //                          staging[1-r] on a receive stream of the peer. Nothing is enqueued
    //                          on the ranks' own streams except one event record each.
    //   sum_finish(lane, ec)   rank r's stream waits for both receive streams' copies of the lane
    //                          (its inbound data; the peer's read of rank r's host buffer, which
    //                          follows rank r's read of buffer[r]) and combines buffer[r] +=
    //                          staging[r].
    //
    // Up to kSumLanes calls may be open at once, one per lane, on disjoint buffer and staging
    // ranges; work enqueued between a lane's begin and finish must neither read nor write that
    // lane's buffer or staging. Calls on a lane alternate begin/finish. Each lane's completion is
    // its own event recorded at begin, so finishing one lane never waits for a later lane. After
    // sum_finish, rank r's stream is ordered after every copy of the lane (the post-condition of
    // allreduce_sum()), so staged and lane calls interleave freely. Eager only (the caller checks
    // that neither stream is capturing); the lanes' pinned buffers are their own, grown on demand
    // (growing synchronizes both devices). Eight lanes: the 27B prefill overlap opens two (one per
    // half-chunk); the sliced prefill (v07/prefill-slices) one per column slice, up to eight.
    static constexpr int kSumLanes = 8;
    void sum_begin(int lane, const std::array<Tensor, 2>& buffer,
                   const std::array<Tensor, 2>& staging, const ExecutionContext& ec);
    void sum_finish(int lane, const ExecutionContext& ec);

    [[nodiscard]] bool sum_open(int lane) const noexcept {
        return lane >= 0 && lane < kSumLanes && lanes_[static_cast<std::size_t>(lane)].active;
    }

private:
    void reserve(std::size_t bytes, std::size_t chunks, const ExecutionContext& ec);
    static void grow(std::vector<cudaEvent_t>& events, std::size_t count, int device);

    std::array<int, 2> devices_{-1, -1};
    std::size_t chunk_bytes_ = 0;
    std::size_t capacity_    = 0;
    std::array<std::byte*, 2> host_{nullptr, nullptr};
    std::array<cudaStream_t, 2> side_{nullptr, nullptr};
    std::array<std::vector<cudaEvent_t>, 2> chunk_done_;

    // T62 state.
    struct Pending {
        bool active                = false;
        const ExecutionContext* ec = nullptr;
        std::array<Tensor, 2> partial, staging, out;
        bool has_out      = false;
        Payload payload   = Payload::FP32;
        std::int64_t rows = 0, columns = 0;
        std::array<std::int64_t, 2> done{0, 0};
        std::array<std::size_t, 2> groups{0, 0}, chunks{0, 0};
    };

    Pending pending_;
    std::int64_t calls_ = 0;
    std::array<cudaStream_t, 2> receive_{nullptr, nullptr};
    // On a pair without peer access (unless NINFER_TP_XIN_KERNEL=0): the T62 receive leg is read by
    // a kernel.
    bool receive_by_kernel_ = false;
    std::array<cudaEvent_t, 2> begun_{nullptr, nullptr};
    std::array<cudaEvent_t, 2> received_{nullptr, nullptr};
    std::array<std::vector<cudaEvent_t>, 2> group_ready_, pipe_chunk_;
    std::array<void*, 2> stats_{nullptr, nullptr};

    // Split-phase lanes. The send/receive streams are shared by the lanes (their calls are
    // issued in time order); everything else is per lane.
    struct Lane {
        bool active          = false;
        std::size_t capacity = 0;
        std::array<Tensor, 2> buffer, staging;
        std::array<std::byte*, 2> host{nullptr, nullptr};
        std::array<cudaEvent_t, 2> begun{nullptr, nullptr};
        std::array<cudaEvent_t, 2> received{nullptr, nullptr};
        std::array<std::vector<cudaEvent_t>, 2> chunk_done;
    };

    void reserve_lane(Lane& lane, std::size_t bytes, std::size_t chunks,
                      const ExecutionContext& ec);
    std::array<Lane, kSumLanes> lanes_;
    std::array<cudaStream_t, 2> lane_send_{nullptr, nullptr};
    std::array<cudaStream_t, 2> lane_receive_{nullptr, nullptr};
};

/**
 * Split-phase BF16 summing all-reduce over a direct cross-device copy (boards WITH peer access).
 *
 * The HostExchange lanes' contract (sum_begin / sum_finish, kSumLanes independent lanes, same
 * arguments, same bits) with the one-call staged path's transport: rank r pulls the peer's operand
 * into staging[r] with the same cudaMemcpyAsync(DeviceToDevice) over UVA that allreduce_sum issues
 * (a direct PCIe copy with peer access), only on a side stream of rank r instead of rank r's own
 * stream, so the ranks keep computing while the bytes move:
 *
 *   sum_begin(lane, ...)  phase A, both ranks: record(begun[r]) on ec.dev[r]->stream (operand r is
 *                         complete, staging[r] is free); phase B, both ranks: the side stream of
 *                         rank r waits begun[r] and begun[1-r], copies buffer[1-r] -> staging[r]
 *                         and records received[r]. Nothing else lands on the ranks' streams.
 *   sum_finish(lane, ec)  rank r's stream waits received[r] (inbound data) and received[1-r] (the
 *                         peer finished reading buffer[r]) and combines buffer[r] += staging[r]
 *                         with residual_add, the one-call combine.
 *
 * Same post-condition as allreduce_sum(): after sum_finish rank r's stream is ordered after the
 * peer's read of buffer[r], so one-call, HostExchange-lane and peer-lane calls interleave freely.
 * Each lane's events are its own: finishing one lane never waits for the other. Eager only.
 *
 * `direct == false` (NINFER_TP_PEER_OVERLAP=force without P2P, a correctness test only): the copy
 * is staged by the driver, which blocks the host and is not ordered after the source device's work
 * (T65), so sum_begin first waits on the host for both begun events and synchronizes the side
 * stream after the copy. Nothing overlaps then; the bits must be the same.
 */
class PeerLanes {
public:
    static constexpr int kSumLanes = HostExchange::kSumLanes;

    PeerLanes(const ExecutionContext& ec, bool direct);
    ~PeerLanes();
    PeerLanes(const PeerLanes&)            = delete;
    PeerLanes& operator=(const PeerLanes&) = delete;

    [[nodiscard]] bool serves(const ExecutionContext& ec) const noexcept;

    [[nodiscard]] bool direct() const noexcept { return direct_; }

    void sum_begin(int lane, const std::array<Tensor, 2>& buffer,
                   const std::array<Tensor, 2>& staging, const ExecutionContext& ec);
    void sum_finish(int lane, const ExecutionContext& ec);

    [[nodiscard]] bool sum_open(int lane) const noexcept {
        return lane >= 0 && lane < kSumLanes && lanes_[static_cast<std::size_t>(lane)].active;
    }

private:
    struct Lane {
        bool active = false;
        std::array<Tensor, 2> buffer, staging;
        std::array<cudaEvent_t, 2> begun{nullptr, nullptr};
        std::array<cudaEvent_t, 2> received{nullptr, nullptr};
    };

    std::array<int, 2> devices_{-1, -1};
    bool direct_ = true;
    // Rank r's side stream (on dev r), shared by the lanes: their calls are issued in time order.
    std::array<cudaStream_t, 2> pull_{nullptr, nullptr};
    std::array<Lane, kSumLanes> lanes_;
};

// T68: the HostExchange attached to `events` when it serves `ec` and neither rank's stream is
// capturing (the eager collectives' own selection); null otherwise (callers keep their UVA copy).
HostExchange* eager_exchange(const ExecutionContext& ec, const PeerEvents& events);

/**
 * Two-device summing all-reduce, in place:
 *
 *   ideal[i] = buffer_rank0[i] + buffer_rank1[i]   for every i,
 *
 * written back to both `buffer[0]` and `buffer[1]`, which afterwards hold the identical sum.
 *
 * `buffer[r]` is a contiguous BF16 tensor resident on `ec.dev[r]`; both ranks carry the same
 * shape. `staging[r]` is scratch of the same dtype and shape, also resident on `ec.dev[r]`, and
 * must not overlap `buffer[r]`; it receives the peer's contribution and its contents after the
 * call are unspecified. Rank r's stream is the only stream that ever touches `staging[r]`.
 *
 * The oracle evaluates `ideal` in FP64 from the represented BF16 inputs; the observable BF16
 * output is that value rounded once to BF16 storage, and the local combine is the qualified
 * residual_add computation body (FP32 accumulation of the two BF16 operands, one
 * round-to-nearest-even on store). The Op holds no persistent state and allocates nothing.
 *
 * Requires `ec.tp == 2` and a live `events`. Consecutive calls sharing the same arguments need no
 * host synchronization between them.
 */
void allreduce_sum(const std::array<Tensor, 2>& buffer, const std::array<Tensor, 2>& staging,
                   const ExecutionContext& ec, const PeerEvents& events);

/**
 * Split-phase allreduce_sum() on `lane` of the eager PeerLanes attached to `events` when there are
 * any (PeerLanes::sum_begin / sum_finish), else of the eager HostExchange (HostExchange::sum_begin
 * / sum_finish): identical arguments, result and bits, but the transfer runs on the transport's
 * own streams between the two calls, so work the caller enqueues in between overlaps it.
 * allreduce_sum_split_available() tells whether the split form can run (eager PeerLanes or an
 * eager HostExchange serving `ec` is attached and neither stream is capturing); begin throws
 * otherwise. The caller completes every begun lane with allreduce_sum_finish() before reading
 * `buffer` or reusing `staging`, and does not touch either in between.
 */
[[nodiscard]] bool allreduce_sum_split_available(const ExecutionContext& ec,
                                                 const PeerEvents& events);
void allreduce_sum_begin(const std::array<Tensor, 2>& buffer, const std::array<Tensor, 2>& staging,
                         const ExecutionContext& ec, const PeerEvents& events, int lane);
void allreduce_sum_finish(const ExecutionContext& ec, const PeerEvents& events, int lane);

/**
 * Two-device summing all-reduce of FP32 partials, rounded once:
 *
 *   sum[i]       = fl32(partial_rank0[i] + partial_rank1[i])     (one IEEE FP32 addition)
 *   partial[r]   = sum                                            (in place, FP32, both ranks)
 *   bf16_out[r]  = RNE_bf16(sum)                                  (when `bf16_out` is non-null)
 *
 * The addition is commutative, so both ranks hold the same bits. This is the exchange for
 * row-parallel partials accumulated in FP32 (the Qwen4Exp MoE: a rank's resident routed experts,
 * its CPU-path experts and, on its blocks, the shared expert, summed in FP32 on that rank), whose
 * block output is then rounded to BF16 exactly once, after the cross-rank sum.
 *
 * `partial[r]`, `staging[r]` are contiguous FP32 tensors of one shape on `ec.dev[r]`;
 * `staging[r]` must not overlap `partial[r]` and its contents afterwards are unspecified.
 * `bf16_out`, when given, holds a contiguous BF16 tensor of the same shape per rank (16-byte
 * aligned), not overlapping the others. Transports as allreduce_sum(): inside a two-device capture
 * with a pipelined PeerMailbox attached and the payload (4 bytes per element) whole 16-byte
 * vectors within one slot, one FP32 exchange kernel per device; otherwise the staged copies and a
 * local FP32 combine. Both transports give identical bits. A mailbox created with the original
 * (Legacy) kernel has no FP32 exchange, so such pairs stay staged.
 *
 * Requires `ec.tp == 2` and a live `events`. Consecutive calls sharing the same arguments need no
 * host synchronization between them.
 */
void allreduce_sum_f32(const std::array<Tensor, 2>& partial, const std::array<Tensor, 2>& staging,
                       const std::array<Tensor, 2>* bf16_out, const ExecutionContext& ec,
                       const PeerEvents& events);

namespace detail {
// The mailbox attached to `events` when a captured exchange of `bytes` would take it: a mailbox
// that serves `ec` is attached, no StagedScope is open, `bytes` is whole 16-byte vectors within one
// slot, and both ranks' streams are capturing into the same capture. Null otherwise. Leaves the
// current device changed (the caller restores it). For fused Ops that exchange through
// PeerMailbox::exchange_slot themselves and must pick the same transport an allreduce would.
[[nodiscard]] PeerMailbox* captured_mailbox_for(std::size_t bytes, const ExecutionContext& ec,
                                                const PeerEvents& events);

// allreduce_sum()'s argument checks alone; allreduce_sum() runs them itself. A row-parallel form
// calls this before issuing either rank's partial, so a call rejected for its all-reduce
// arguments enqueues nothing.
void require_allreduce_sum_arguments(const std::array<Tensor, 2>& buffer,
                                     const std::array<Tensor, 2>& staging,
                                     const ExecutionContext& ec, const PeerEvents& events);
} // namespace detail

/**
 * Two-device row gather, exact (no arithmetic). The gathered axis is `ne[1]`, so each rank
 * contributes one contiguous block of a `[C, R]` tensor (`ne[0] == C` is the row length,
 * `ne[1] == R` the row count, `ne[2] == ne[3] == 1`):
 *
 *   destination[r][c, row] = part[0][c, row]                    for row <  part[0].ne[1]
 *   destination[r][c, row] = part[1][c, row - part[0].ne[1]]    otherwise
 *
 * for both r, so each device ends up holding the identical full image. `part[r]` is the row range
 * owned by `ec.dev[r]`: rank 0 owns the leading rows and rank 1 the trailing rows, and the two
 * counts must sum to the destination row count. `destination[r]` and `part[r]` are contiguous,
 * share one dtype, agree on `ne[0]`, live on `ec.dev[r]`, and must not overlap. A caller whose
 * split axis is not `ne[1]` (for example `[V, B]` logits split by vocabulary with B > 1) arranges
 * the layout so that it is; this Op relocates contiguous storage and performs no transpose.
 *
 * `destination[r]` is written only by rank r's stream: rank r copies its own block locally and
 * pulls the peer's block. The transformation is a pure relocation of storage, so it is verified by
 * exact byte comparison. The Op holds no persistent state and allocates nothing.
 *
 * Requires `ec.tp == 2` and a live `events`. Consecutive calls sharing the same arguments need no
 * host synchronization between them.
 *
 * Test-only, kept for symmetry with allreduce_sum: no model path calls it since the two-device
 * logits are assembled on rank 0 alone (output_logits_split_rank0 in
 * src/models/qwen3_5/execution/tp.h); the all-reduce and output-head split suites exercise it.
 */
void allgather_rows(const std::array<Tensor, 2>& destination, const std::array<Tensor, 2>& part,
                    const ExecutionContext& ec, const PeerEvents& events);

} // namespace ninfer::ops
