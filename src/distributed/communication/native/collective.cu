#include "distributed/communication/native/protocol.h"

#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <algorithm>

namespace ninfer::distributed::detail::native {
namespace {

__device__ std::uint64_t acquire(Signal& value) {
    return cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system>(value.value)
        .load(cuda::memory_order_acquire);
}

__device__ void release(Signal& value, std::uint64_t next) {
    cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system>(value.value)
        .store(next, cuda::memory_order_release);
}

__device__ bool stopped(Control& control) {
    return cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system>(control.abort.value)
               .load(cuda::memory_order_relaxed) != 0;
}

__device__ bool wait(Signal& signal, std::uint64_t sequence, Control& control) {
    auto flag      = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system>(signal.value);
    unsigned spins = 0;
    while (flag.load(cuda::memory_order_relaxed) < sequence) {
        if (!(spins++ & 31U) && stopped(control)) return false;
        __nanosleep(32);
    }
    (void)flag.load(cuda::memory_order_acquire);
    return true;
}

__device__ uint4 load(const std::byte* input, unsigned bytes) {
    if (bytes == 16 && reinterpret_cast<std::uintptr_t>(input) % 16 == 0)
        return *reinterpret_cast<const uint4*>(input);
    uint4 result{};
    if (reinterpret_cast<std::uintptr_t>(input) % 4 == 0 && bytes % 4 == 0) {
#pragma unroll
        for (unsigned i = 0; i < 4; ++i)
            if (i * 4 < bytes)
                reinterpret_cast<unsigned*>(&result)[i] =
                    reinterpret_cast<const unsigned*>(input)[i];
    } else {
#pragma unroll
        for (unsigned i = 0; i < 16; ++i)
            if (i < bytes) reinterpret_cast<std::byte*>(&result)[i] = input[i];
    }
    return result;
}

__device__ void store(std::byte* output, uint4 value, unsigned bytes) {
    if (bytes == 16 && reinterpret_cast<std::uintptr_t>(output) % 16 == 0) {
        *reinterpret_cast<uint4*>(output) = value;
    } else if (reinterpret_cast<std::uintptr_t>(output) % 4 == 0 && bytes % 4 == 0) {
#pragma unroll
        for (unsigned i = 0; i < 4; ++i)
            if (i * 4 < bytes)
                reinterpret_cast<unsigned*>(output)[i] =
                    reinterpret_cast<const unsigned*>(&value)[i];
    } else {
#pragma unroll
        for (unsigned i = 0; i < 16; ++i)
            if (i < bytes) output[i] = reinterpret_cast<const std::byte*>(&value)[i];
    }
}

template <class T>
__device__ T combine(T a, T b, unsigned reduction) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
        return reduction == static_cast<unsigned>(Reduction::Sum) ? __hadd(a, b) : __hmax(a, b);
    else if constexpr (std::is_same_v<T, __half>)
        return reduction == static_cast<unsigned>(Reduction::Sum) ? __hadd(a, b) : __hmax(a, b);
    else if constexpr (std::is_same_v<T, float>)
        return reduction == static_cast<unsigned>(Reduction::Sum) ? a + b : fmaxf(a, b);
    else {
        if (reduction == static_cast<unsigned>(Reduction::Max)) return a > b ? a : b;
        using U       = std::make_unsigned_t<T>;
        const U value = static_cast<U>(a) + static_cast<U>(b);
        return __builtin_bit_cast(T, value);
    }
}

template <class T>
__device__ uint4 reduce(uint4 a, uint4 b, unsigned reduction) {
    uint4 result;
    if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, __half>) {
        using Pair = std::conditional_t<std::is_same_v<T, __half>, __half2, __nv_bfloat162>;
#pragma unroll
        for (unsigned i = 0; i < 4; ++i) {
            const auto x = reinterpret_cast<const Pair*>(&a)[i];
            const auto y = reinterpret_cast<const Pair*>(&b)[i];
            reinterpret_cast<Pair*>(&result)[i] =
                reduction == static_cast<unsigned>(Reduction::Sum) ? __hadd2(x, y) : __hmax2(x, y);
        }
    } else {
#pragma unroll
        for (unsigned i = 0; i < 16 / sizeof(T); ++i)
            reinterpret_cast<T*>(&result)[i] = combine(
                reinterpret_cast<const T*>(&a)[i], reinterpret_cast<const T*>(&b)[i], reduction);
    }
    return result;
}

template <class T, bool Reduce, unsigned Threads, unsigned Vectors, bool Bulk>
__global__ __launch_bounds__(Threads) void transfer(const __grid_constant__ Operation op) {
    const bool separate = Bulk && op.kind == kPeerOperation;
    const bool tx_lane  = !separate || blockIdx.x < op.tx.lanes;
    const bool rx_lane  = !separate || blockIdx.x >= op.tx.lanes;
    const unsigned lane = separate && rx_lane ? blockIdx.x - op.tx.lanes : blockIdx.x;
    const auto epoch    = Bulk ? op.activity_state->epoch + 1 : 0;
    if constexpr (Bulk)
        if (!blockIdx.x && !threadIdx.x) release(op.activity->issued, epoch);
    auto tx_sequence    = tx_lane && lane < op.tx.lanes ? op.send.state->send[lane] : 0;
    auto rx_sequence    = rx_lane && lane < op.rx.lanes ? op.receive.state->receive[lane] : 0;
    const auto tx_steps = tx_lane && lane < op.tx.lanes && lane < op.tx.tiles
                              ? 1 + (op.tx.tiles - 1 - lane) / op.tx.lanes
                              : 0;
    const auto rx_steps = rx_lane && lane < op.rx.lanes && lane < op.rx.tiles
                              ? 1 + (op.rx.tiles - 1 - lane) / op.rx.lanes
                              : 0;
    __shared__ bool ready;
    for (unsigned step = 0; step < max(tx_steps, rx_steps); ++step) {
        const bool sends = step < tx_steps, receives = step < rx_steps;
        const std::size_t tx_offset = (std::size_t(step) * op.tx.lanes + lane) * op.tx.tile_bytes;
        const std::size_t rx_offset = (std::size_t(step) * op.rx.lanes + lane) * op.rx.tile_bytes;
        const auto tx_bytes = sends ? static_cast<unsigned>(min(std::size_t(op.tx.tile_bytes),
                                                                op.send_bytes - tx_offset))
                                    : 0;
        const auto rx_bytes = receives ? static_cast<unsigned>(min(std::size_t(op.rx.tile_bytes),
                                                                   op.receive_bytes - rx_offset))
                                       : 0;
        if (sends) ++tx_sequence;
        if (receives) ++rx_sequence;
        const auto tx_slot = tx_sequence % kSlots, rx_slot = rx_sequence % kSlots;
        if (!threadIdx.x) {
            ready = (!sends || !stopped(*op.send.control)) &&
                    (!receives || (sends && op.send.control == op.receive.control) ||
                     !stopped(*op.receive.control));
            if (ready && sends && tx_sequence > kSlots)
                ready = wait(op.send.control->reusable[tx_slot][lane], tx_sequence - kSlots,
                             *op.send.control);
            if (receives) release(op.receive.control->expected[lane], rx_sequence);
        }
        __syncthreads();
        if (!ready) return;
        auto* tx = sends ? reinterpret_cast<Packet*>(op.send.send + tx_slot * op.send.slot_stride +
                                                     lane * op.send.packet_stride)
                         : nullptr;
        const auto* rx = receives ? reinterpret_cast<const Packet*>(
                                        op.receive.receive + rx_slot * op.receive.slot_stride +
                                        lane * op.receive.packet_stride)
                                  : nullptr;
        uint4 local[Reduce ? Vectors : 1];
#pragma unroll
        for (unsigned v = 0; v < Vectors; ++v) {
            const auto position = (v * Threads + threadIdx.x) * 16;
            if constexpr (Reduce)
                if (position < rx_bytes)
                    local[v] = load(op.local + rx_offset + position, min(16U, rx_bytes - position));
            if (position < tx_bytes) {
                uint4 value;
                if constexpr (Reduce) {
                    value =
                        op.input == op.local && tx_offset == rx_offset
                            ? local[v]
                            : load(op.input + tx_offset + position, min(16U, tx_bytes - position));
                } else
                    value = load(op.input + tx_offset + position, min(16U, tx_bytes - position));
                store(reinterpret_cast<std::byte*>(tx) + kHeaderBytes + position, value,
                      min(16U, tx_bytes - position));
                if (op.local_copy && op.local_copy != op.input)
                    store(op.local_copy + tx_offset + position, value,
                          min(16U, tx_bytes - position));
            }
        }
        if (sends) {
            if (!threadIdx.x)
                *tx = {tx_sequence, op.send_bytes, tx_offset,    tx_bytes,
                       op.kind,     op.dtype,      op.reduction, op.root};
            // The barrier orders all writers before the leader's cumulative system release.
            __syncthreads();
            if (!threadIdx.x) release(op.send.control->ready[tx_slot][lane], tx_sequence);
        }
        if (receives) {
            if (!threadIdx.x) {
                ready = wait(op.receive.control->received[rx_slot][lane], rx_sequence,
                             *op.receive.control);
                if (ready && (rx->sequence != rx_sequence || rx->total_bytes != op.receive_bytes ||
                              rx->offset != rx_offset || rx->bytes != rx_bytes ||
                              rx->operation != op.kind || rx->dtype != op.dtype ||
                              rx->reduction != op.reduction || rx->root != op.root)) {
                    release(op.receive.control->abort, 2);
                    ready = false;
                }
            }
            __syncthreads();
            if (!ready) return;
#pragma unroll
            for (unsigned v = 0; v < Vectors; ++v) {
                const auto position = (v * Threads + threadIdx.x) * 16;
                if (position < rx_bytes) {
                    auto value =
                        load(reinterpret_cast<const std::byte*>(rx) + kHeaderBytes + position,
                             min(16U, rx_bytes - position));
                    if constexpr (Reduce)
                        value = op.rank == 0 ? reduce<T>(local[v], value, op.reduction)
                                             : reduce<T>(value, local[v], op.reduction);
                    store(op.output + rx_offset + position, value, min(16U, rx_bytes - position));
                }
            }
            // Finish every reader before returning the RX slot to the remote producer.
            __syncthreads();
            if (!threadIdx.x) release(op.receive.control->consumed[rx_slot][lane], rx_sequence);
        }
    }
    if (!threadIdx.x) {
        if (tx_lane && lane < op.tx.lanes) op.send.state->send[lane] = tx_sequence;
        if (rx_lane && lane < op.rx.lanes) op.receive.state->receive[lane] = rx_sequence;
        if constexpr (Bulk) {
            if (atomicAdd(&op.activity_state->completed, 1) == gridDim.x - 1) {
                op.activity_state->completed = 0;
                op.activity_state->epoch     = epoch;
                release(op.activity->completed, epoch);
            }
        }
    }
}

template <class T, bool Reduce, unsigned Vectors>
void submit_bulk(const Operation& op, unsigned lanes, CudaExecutionView execution) {
    auto function = transfer<T, Reduce, 512, Vectors, true>;
    void* args[]  = {const_cast<Operation*>(&op)};
    CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(function), dim3(lanes),
                                           dim3(512), args, 0, execution.stream()));
}

template <class T, bool Reduce>
void submit(const Operation& op, CudaExecutionView execution) {
    const auto lanes = op.activity && op.kind == kPeerOperation
                           ? op.tx.lanes + op.rx.lanes
                           : std::max(op.tx.lanes, op.rx.lanes);
    if (op.activity) {
        const auto tile = std::max(op.tx.tile_bytes, op.rx.tile_bytes);
        if (tile <= 16384)
            submit_bulk<T, Reduce, 2>(op, lanes, execution);
        else if (tile <= 65536)
            submit_bulk<T, Reduce, 8>(op, lanes, execution);
        else
            submit_bulk<T, Reduce, 16>(op, lanes, execution);
    } else if (std::max(op.send_bytes, op.receive_bytes) >= 4096) {
        transfer<T, Reduce, 512, 1, false><<<1, 512, 0, execution.stream()>>>(op);
    } else {
        transfer<T, Reduce, 256, 2, false><<<1, 256, 0, execution.stream()>>>(op);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch(const Operation& op, CudaExecutionView execution) {
    const bool reduces = op.kind == static_cast<unsigned>(Collective::AllReduce) ||
                         op.kind == static_cast<unsigned>(Collective::ReduceScatter);
    if (!reduces) return submit<std::uint8_t, false>(op, execution);
    switch (static_cast<DType>(op.dtype)) {
    case DType::BF16:
        return submit<__nv_bfloat16, true>(op, execution);
    case DType::FP16:
        return submit<__half, true>(op, execution);
    case DType::FP32:
        return submit<float, true>(op, execution);
    case DType::I8:
        return submit<std::int8_t, true>(op, execution);
    case DType::U8:
        return submit<std::uint8_t, true>(op, execution);
    case DType::I32:
        return submit<std::int32_t, true>(op, execution);
    case DType::U32:
        return submit<std::uint32_t, true>(op, execution);
    case DType::I64:
        return submit<std::int64_t, true>(op, execution);
    case DType::U64:
        return submit<std::uint64_t, true>(op, execution);
    default:
        throw std::invalid_argument("native collective dtype is not supported");
    }
}

} // namespace ninfer::distributed::detail::native
