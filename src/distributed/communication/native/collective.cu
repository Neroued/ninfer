#include "distributed/communication/native/protocol.h"

#include <cuda/atomic>
#include <cuda_bf16.h>

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

__device__ bool wait(Signal& value, std::uint64_t sequence, Control& control) {
    auto flag      = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system>(value.value);
    unsigned spins = 0;
    while (flag.load(cuda::memory_order_relaxed) < sequence) {
        if (!(spins++ & 31U) && acquire(control.abort)) return false;
        __nanosleep(32);
    }
    (void)flag.load(cuda::memory_order_acquire);
    return !acquire(control.abort);
}

__device__ bool reserve(DeviceView view, std::uint64_t sequence) {
    if (acquire(view.control->abort)) return false;
    if (sequence <= kSlots) return true;
    const unsigned slot = sequence % kSlots;
    return wait(view.control->reusable[slot], sequence - kSlots, *view.control);
}

__device__ void copy_bytes(const std::byte* input, std::byte* output, unsigned bytes,
                           unsigned worker, unsigned workers) {
    if ((reinterpret_cast<std::uintptr_t>(input) | reinterpret_cast<std::uintptr_t>(output)) % 16 ==
        0) {
        const auto vectors = bytes / 16;
        for (unsigned i = worker; i < vectors; i += workers)
            reinterpret_cast<uint4*>(output)[i] = reinterpret_cast<const uint4*>(input)[i];
        for (unsigned i = vectors * 16 + worker; i < bytes; i += workers) output[i] = input[i];
    } else {
        for (unsigned i = worker; i < bytes; i += workers) output[i] = input[i];
    }
}

struct Part {
    Packet* send;
    const Packet* receive;
    unsigned offset, bytes;
};

__device__ Part part_view(DeviceView view, std::uint64_t sequence, unsigned bytes, unsigned part,
                          LaunchPlan plan) {
    const auto packet = sequence % kSlots * view.slot_stride + part * plan.packet_stride;
    const auto offset = part * plan.part_bytes;
    return {reinterpret_cast<Packet*>(view.send + packet),
            reinterpret_cast<const Packet*>(view.receive + packet), offset,
            offset < bytes ? min(plan.part_bytes, bytes - offset) : 0};
}

template <bool Single>
__device__ void produce(DeviceView view, const CollectiveCall& call, const void* input,
                        void* output, unsigned bytes, std::uint64_t sequence, unsigned part,
                        LaunchPlan plan, unsigned block) {
    const auto p    = part_view(view, sequence, bytes, part, plan);
    const bool data = call.operation != Collective::Broadcast || view.rank == call.root;
    if (!block && !threadIdx.x) {
        *p.send = {.sequence    = sequence,
                   .bytes       = data ? p.bytes : 0,
                   .total_bytes = bytes,
                   .operation   = static_cast<unsigned>(call.operation),
                   .dtype       = static_cast<unsigned>(call.dtype),
                   .root        = call.root,
                   .parts       = plan.parts,
                   .rails       = plan.rails};
    }
    const auto worker  = block * blockDim.x + threadIdx.x;
    const auto workers = Single ? blockDim.x : plan.blocks_per_part * blockDim.x;
    if (data)
        copy_bytes(static_cast<const std::byte*>(input) + p.offset,
                   reinterpret_cast<std::byte*>(p.send) + kHeaderBytes, p.bytes, worker, workers);
    if (call.operation == Collective::AllGather) {
        auto* local_output = static_cast<std::byte*>(output) + view.rank * bytes + p.offset;
        const auto* source = static_cast<const std::byte*>(input) + p.offset;
        if (source != local_output) copy_bytes(source, local_output, p.bytes, worker, workers);
    }
    if (call.operation == Collective::Broadcast && data && input != output)
        copy_bytes(static_cast<const std::byte*>(input) + p.offset,
                   static_cast<std::byte*>(output) + p.offset, p.bytes, worker, workers);
    __threadfence_system();
    __syncthreads();
    if (!threadIdx.x) {
        if constexpr (Single) {
            release(view.control->ready[sequence % kSlots][part], sequence);
        } else if (atomicAdd(view.state->produced + part, 1) == plan.blocks_per_part - 1) {
            view.state->produced[part] = 0;
            release(view.control->ready[sequence % kSlots][part], sequence);
        }
    }
}

template <DType Type>
__device__ void reduce(const std::byte* a, const std::byte* b, std::byte* output, unsigned bytes,
                       unsigned worker, unsigned workers) {
    if (reinterpret_cast<std::uintptr_t>(output) % 16 == 0) {
        const auto vectors = bytes / 16;
        for (unsigned i = worker; i < vectors; i += workers) {
            const uint4 x = reinterpret_cast<const uint4*>(a)[i];
            const uint4 y = reinterpret_cast<const uint4*>(b)[i];
            uint4 z;
            if constexpr (Type == DType::BF16) {
#pragma unroll
                for (unsigned j = 0; j < 4; ++j)
                    reinterpret_cast<__nv_bfloat162*>(&z)[j] =
                        __hadd2(reinterpret_cast<const __nv_bfloat162*>(&x)[j],
                                reinterpret_cast<const __nv_bfloat162*>(&y)[j]);
            } else if constexpr (Type == DType::U32) {
                z = {x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w};
            } else {
#pragma unroll
                for (unsigned j = 0; j < 2; ++j) {
                    const auto left  = reinterpret_cast<const std::uint64_t*>(&x)[j];
                    const auto right = reinterpret_cast<const std::uint64_t*>(&y)[j];
                    reinterpret_cast<std::uint64_t*>(&z)[j] = left > right ? left : right;
                }
            }
            reinterpret_cast<uint4*>(output)[i] = z;
        }
        a += vectors * 16;
        b += vectors * 16;
        output += vectors * 16;
        bytes -= vectors * 16;
    }
    if constexpr (Type == DType::BF16) {
        const auto* x = reinterpret_cast<const __nv_bfloat162*>(a);
        const auto* y = reinterpret_cast<const __nv_bfloat162*>(b);
        auto* z       = reinterpret_cast<__nv_bfloat162*>(output);
        if (reinterpret_cast<std::uintptr_t>(output) % 4 == 0) {
            for (unsigned i = worker; i < bytes / 4; i += workers) z[i] = __hadd2(x[i], y[i]);
            if (bytes % 4 && worker == 0) {
                const auto i = bytes / 2 - 1;
                reinterpret_cast<__nv_bfloat16*>(output)[i] =
                    __hadd(reinterpret_cast<const __nv_bfloat16*>(a)[i],
                           reinterpret_cast<const __nv_bfloat16*>(b)[i]);
            }
        } else {
            for (unsigned i = worker; i < bytes / 2; i += workers)
                reinterpret_cast<__nv_bfloat16*>(output)[i] =
                    __hadd(reinterpret_cast<const __nv_bfloat16*>(a)[i],
                           reinterpret_cast<const __nv_bfloat16*>(b)[i]);
        }
    } else if constexpr (Type == DType::U32) {
        for (unsigned i = worker; i < bytes / 4; i += workers)
            reinterpret_cast<unsigned*>(output)[i] =
                reinterpret_cast<const unsigned*>(a)[i] + reinterpret_cast<const unsigned*>(b)[i];
    } else {
        for (unsigned i = worker; i < bytes / 8; i += workers) {
            const auto x = reinterpret_cast<const std::uint64_t*>(a)[i];
            const auto y = reinterpret_cast<const std::uint64_t*>(b)[i];
            reinterpret_cast<std::uint64_t*>(output)[i] = x > y ? x : y;
        }
    }
}

template <DType Type, bool Single>
__device__ bool consume(DeviceView view, const CollectiveCall& call, void* output, unsigned bytes,
                        std::uint64_t sequence, unsigned part, LaunchPlan plan, unsigned block) {
    __shared__ bool ready;
    const auto p = part_view(view, sequence, bytes, part, plan);
    if (!threadIdx.x) {
        const bool receives = call.operation != Collective::Broadcast || view.rank != call.root;
        ready               = !receives ||
                wait(view.control->received[sequence % kSlots][part], sequence, *view.control);
        if (receives && ready &&
            (p.receive->sequence != sequence || p.receive->total_bytes != bytes ||
             p.receive->operation != static_cast<unsigned>(call.operation) ||
             p.receive->dtype != static_cast<unsigned>(call.dtype) ||
             p.receive->root != call.root || p.receive->parts != plan.parts)) {
            release(view.control->abort, 2);
            ready = false;
        }
    }
    __syncthreads();
    if (!ready) return false;
    const auto* local  = reinterpret_cast<const std::byte*>(p.send) + kHeaderBytes;
    const auto* peer   = reinterpret_cast<const std::byte*>(p.receive) + kHeaderBytes;
    auto* destination  = static_cast<std::byte*>(output) + p.offset;
    const auto worker  = block * blockDim.x + threadIdx.x,
               workers = Single ? blockDim.x : plan.blocks_per_part * blockDim.x;
    if (call.operation == Collective::AllReduce) {
        reduce<Type>(view.rank == 0 ? local : peer, view.rank == 0 ? peer : local, destination,
                     p.bytes, worker, workers);
    } else if (call.operation == Collective::AllGather) {
        copy_bytes(peer, destination + (1 - view.rank) * bytes, p.bytes, worker, workers);
    } else if (view.rank != call.root) {
        copy_bytes(peer, destination, p.bytes, worker, workers);
    }
    __threadfence_system();
    __syncthreads();
    if (!threadIdx.x) {
        bool last = true;
        if constexpr (!Single)
            last = atomicAdd(&view.state->completed, 1) == plan.parts * plan.blocks_per_part - 1;
        if (last) {
            if constexpr (!Single) view.state->completed = 0;
            view.state->sequence = sequence;
            release(view.control->consumed[sequence % kSlots], sequence);
        }
    }
    return true;
}

__global__ void pack(DeviceView view, CollectiveCall call, const void* input, void* output,
                     unsigned bytes, LaunchPlan plan) {
    __shared__ bool usable;
    const auto sequence = view.state->sequence + 1;
    if (!threadIdx.x) usable = reserve(view, sequence);
    __syncthreads();
    if (!usable) return;
    produce<false>(view, call, input, output, bytes, sequence, blockIdx.x / plan.blocks_per_part,
                   plan, blockIdx.x % plan.blocks_per_part);
}

template <DType Type>
__global__ void finish(DeviceView view, CollectiveCall call, void* output, unsigned bytes,
                       LaunchPlan plan) {
    const auto sequence = view.state->sequence + 1;
    consume<Type, false>(view, call, output, bytes, sequence, blockIdx.x / plan.blocks_per_part,
                         plan, blockIdx.x % plan.blocks_per_part);
}

template <DType Type>
__global__ void exchange(DeviceView view, CollectiveCall call, const void* input, void* output,
                         unsigned bytes, LaunchPlan plan) {
    __shared__ bool usable;
    const auto sequence = view.state->sequence + 1;
    if (!threadIdx.x) usable = reserve(view, sequence);
    __syncthreads();
    if (!usable) return;
    produce<true>(view, call, input, output, bytes, sequence, 0, plan, 0);
    consume<Type, true>(view, call, output, bytes, sequence, 0, plan, 0);
}

template <DType Type>
void launch(DeviceView view, const CollectiveCall& call, const void* input, void* output,
            unsigned bytes, LaunchPlan plan, CudaExecutionView execution) {
    if (plan.fused) {
        exchange<Type>
            <<<1, plan.threads, 0, execution.stream()>>>(view, call, input, output, bytes, plan);
    } else {
        const auto blocks = plan.parts * plan.blocks_per_part;
        pack<<<blocks, 256, 0, execution.stream()>>>(view, call, input, output, bytes, plan);
        finish<Type><<<blocks, 256, 0, execution.stream()>>>(view, call, output, bytes, plan);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_collective(DeviceView view, const CollectiveCall& call, const void* source,
                       void* destination, LaunchPlan plan, CudaExecutionView execution) {
    const auto bytes   = static_cast<unsigned>(call.count * dtype_size(call.dtype));
    plan.packet_stride = packet_offset(view.capacity, 1, plan.parts);
    plan.part_bytes    = part_size(bytes, plan.parts);
    if (call.operation != Collective::AllReduce || call.dtype == DType::BF16)
        launch<DType::BF16>(view, call, source, destination, bytes, plan, execution);
    else if (call.dtype == DType::U32)
        launch<DType::U32>(view, call, source, destination, bytes, plan, execution);
    else
        launch<DType::U64>(view, call, source, destination, bytes, plan, execution);
}

} // namespace ninfer::distributed::detail::native
