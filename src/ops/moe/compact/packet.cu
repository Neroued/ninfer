#include "ninfer/ops/moe_exchange.h"
#include "core/device.h"
#include "ops/common/host/operands.h"
#include <algorithm>
#include <cub/block/block_scan.cuh>
#include <cuda_bf16.h>

namespace ninfer::ops {
namespace {
void validity_view(MoeTokenValidity v, int columns, Device device) {
    if (!v.valid_columns.data) return;
    if (v.tokens_per_row <= 0 || columns % v.tokens_per_row)
        throw std::invalid_argument("expert block validity extent differs");
    common::operand::tensor(v.valid_columns, DType::I32, {columns / v.tokens_per_row}, 4,
                            "expert dispatch", "valid rows", device);
}

__global__ void validity(const int* lengths, int width, int base, int n, int* flags) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) flags[i] = (base + i) % width < lengths[(base + i) / width];
}

__global__ void reset(ExpertPeerBuffers* peers, int columns) {
    const auto peer = peers[blockIdx.y];
    const int col   = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < columns) peer.source_to_row[col] = -1;
    if (col < 2) peer.send.counts[col] = 0;
}

__global__ void index_rows(const int* ids, const float* weights, int k, int columns,
                           const int* valid, int width, ExpertDispatchView map,
                           ExpertPeerBuffers* peers) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= columns || (valid && col % width >= valid[col / width])) return;
    for (int j = 0; j < k; ++j) {
        const auto a     = std::int64_t(col) * k + j;
        const int expert = ids[a];
        if (expert < 0) continue;
        for (int t = map.offsets[expert]; t < map.offsets[expert + 1]; ++t) {
            const auto dest = map.destinations[t];
            const auto peer = peers[dest.peer];
            int& row        = peer.source_to_row[col];
            if (row < 0) row = atomicAdd(peer.send.counts, 1);
            const int hit       = atomicAdd(peer.send.counts + 1, 1);
            peer.send.hits[hit] = {row, dest.slot, weights[a]};
        }
    }
}

__global__ void copy_rows(const std::uint16_t* hidden, int h, ExpertPeerBuffers* peers) {
    const auto p  = peers[blockIdx.y];
    const int row = p.source_to_row[blockIdx.x];
    if (row < 0) return;
    const auto* src = reinterpret_cast<const uint4*>(hidden + std::int64_t(blockIdx.x) * h);
    auto* dst       = reinterpret_cast<uint4*>(p.send.hidden + std::int64_t(row) * h);
    for (int i = threadIdx.x; i < h / 8; i += blockDim.x) dst[i] = src[i];
}

__global__ void histogram(ExpertPacketBatch batch, int* counts, int* totals) {
    const auto p = batch.peers[blockIdx.y].receive;
    if (!blockIdx.x && !threadIdx.x) {
        atomicAdd(totals, p.counts[0]);
        atomicAdd(totals + 1, p.counts[1]);
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < p.counts[1];
         i += gridDim.x * blockDim.x)
        atomicAdd(counts + p.hits[i].slot, 1);
}

__global__ void scan_slots(int* counts, int* offsets, int slots) {
    using Scan = cub::BlockScan<int, 128>;
    __shared__ typename Scan::TempStorage scratch;
    int base = 0;
    for (int start = 0; start < slots; start += 128) {
        const int slot = start + threadIdx.x;
        int prefix, total;
        Scan(scratch).ExclusiveSum(slot < slots ? counts[slot] : 0, prefix, total);
        if (slot < slots) offsets[slot] = counts[slot] = base + prefix;
        base += total;
        __syncthreads();
    }
    if (!threadIdx.x) offsets[slots] = base;
}

__global__ void scatter(ExpertPacketBatch batch, int* cursors, int* rows, float* coefficients) {
    const auto p = batch.peers[blockIdx.y].receive;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < p.counts[1];
         i += gridDim.x * blockDim.x) {
        const auto hit       = p.hits[i];
        const int target     = atomicAdd(cursors + hit.slot, 1);
        rows[target]         = blockIdx.y * batch.columns + hit.row;
        coefficients[target] = hit.coefficient;
    }
}

__global__ void combine(const ExpertPeerBuffers* peers, int peer_count, const float* shared,
                        __nv_bfloat16* out, int h, const int* lengths, int width, bool add,
                        const float* gate) {
    const int col   = blockIdx.y;
    const bool live = !lengths || col % width < lengths[col / width];
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < h; r += gridDim.x * blockDim.x) {
        const auto i = std::int64_t(col) * h + r;
        if (!live) {
            if (!add) out[i] = __float2bfloat16(0);
            continue;
        }
        float sum = gate ? shared[i] * gate[col] : shared[i];
        for (int d = 0; d < peer_count; ++d) {
            const auto p = peers[d];
            const int u  = p.source_to_row[col];
            if (u >= 0) sum += p.returned[std::int64_t(u) * h + r];
        }
        if (add) sum += __bfloat162float(out[i]);
        out[i] = __float2bfloat16(sum);
    }
}

__global__ void observe(const ExpertPeerBuffers* peers, int n, std::uint64_t* totals) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < n) {
        const auto p = peers[d];
        for (int i = 0; i < 2; ++i) {
            totals[d * 4 + i] += p.send.counts[i];
            totals[d * 4 + i + 2] += p.receive.counts[i];
        }
    }
}
} // namespace

void expert_block_validity(MoeTokenValidity input, int base, Tensor& flags,
                           CudaExecutionView exec) {
    const int n = flags.ne[0], rows = input.valid_columns.ne[0];
    if (base < 0 || n <= 0 || std::int64_t(base) + n > std::int64_t(rows) * input.tokens_per_row ||
        std::int64_t(base) + n > INT_MAX)
        throw std::invalid_argument("expert validity block exceeds source rows");
    common::operand::tensor(flags, DType::I32, {n}, 4, "expert dispatch", "block flags",
                            exec.device());
    common::operand::tensor(input.valid_columns, DType::I32, {rows}, 4, "expert dispatch",
                            "valid rows", exec.device());
    validity<<<(flags.ne[0] + 127) / 128, 128, 0, exec.stream()>>>(
        static_cast<const int*>(input.valid_columns.data), input.tokens_per_row, base, flags.ne[0],
        static_cast<int*>(flags.data));
    CUDA_CHECK(cudaGetLastError());
}

void pack_expert_rows(const Tensor* x, const ExpertRoutes* routes, MoeTokenValidity validity,
                      ExpertDispatchView map, ExpertPeerBuffers* p, int peers, int capacity,
                      CudaExecutionView exec) {
    if (!p || !map.offsets || !map.destinations || peers <= 0 || capacity <= 0 ||
        bool(x) != bool(routes))
        throw std::invalid_argument("invalid expert packet addresses");
    const auto stream = exec.stream();
    if (x) {
        const int n = x->ne[1], h = x->ne[0], k = routes->ids.ne[0];
        if (n > capacity || h % 8)
            throw std::invalid_argument("expert dispatch source exceeds packet geometry");
        common::operand::tensor(*x, DType::BF16, {h, n}, 16, "expert dispatch", "hidden",
                                exec.device());
        common::operand::tensor(routes->ids, DType::I32, {k, n}, 4, "expert dispatch", "route IDs",
                                exec.device());
        common::operand::tensor(routes->coefficients, DType::FP32, {k, n}, 4, "expert dispatch",
                                "route coefficients", exec.device());
        validity_view(validity, n, exec.device());
    }
    reset<<<dim3((capacity - 1) / 128 + 1, peers), 128, 0, stream>>>(p, capacity);
    if (x) {
        const int n = x->ne[1], h = x->ne[0], k = routes->ids.ne[0];
        index_rows<<<(n + 127) / 128, 128, 0, stream>>>(
            static_cast<const int*>(routes->ids.data),
            static_cast<const float*>(routes->coefficients.data), k, n,
            static_cast<const int*>(validity.valid_columns.data), validity.tokens_per_row, map, p);
        copy_rows<<<dim3(n, peers), 128, 0, stream>>>(static_cast<const std::uint16_t*>(x->data), h,
                                                      p);
    }
    CUDA_CHECK(cudaGetLastError());
}

void group_expert_hits(ExpertPacketBatch batch, const CompactExpertAssignments& a, Tensor& cursors,
                       CudaExecutionView exec) {
    const int slots = a.logical_ids.ne[0], capacity = a.rows.ne[0];
    if (!batch.peers || batch.device != exec.device() || batch.segments <= 0 ||
        batch.columns <= 0 || batch.segments > 65535 ||
        std::int64_t(batch.columns) * batch.segments > INT_MAX || slots <= 0 || slots == INT_MAX ||
        capacity <= 0)
        throw std::invalid_argument("invalid compact expert grouping geometry");
    common::operand::tensor(a.counts, DType::I32, {2}, 4, "expert grouping", "counts",
                            exec.device());
    common::operand::tensor(a.offsets, DType::I32, {slots + 1}, 4, "expert grouping", "offsets",
                            exec.device());
    common::operand::tensor(a.rows, DType::I32, {capacity}, 4, "expert grouping", "rows",
                            exec.device());
    common::operand::tensor(a.coefficients, DType::FP32, {capacity}, 4, "expert grouping",
                            "coefficients", exec.device());
    common::operand::tensor(cursors, DType::I32, {slots}, 4, "expert grouping", "cursors",
                            exec.device());
    const int grid = std::min((capacity + 127) / 128, 4 * exec.multiprocessor_count());
    auto* c        = static_cast<int*>(cursors.data);
    CUDA_CHECK(cudaMemsetAsync(c, 0, slots * sizeof(int), exec.stream()));
    CUDA_CHECK(cudaMemsetAsync(a.counts.data, 0, a.counts.bytes(), exec.stream()));
    const dim3 blocks(grid, batch.segments);
    histogram<<<blocks, 128, 0, exec.stream()>>>(batch, c, static_cast<int*>(a.counts.data));
    scan_slots<<<1, 128, 0, exec.stream()>>>(c, static_cast<int*>(a.offsets.data), slots);
    scatter<<<blocks, 128, 0, exec.stream()>>>(batch, c, static_cast<int*>(a.rows.data),
                                               static_cast<float*>(a.coefficients.data));
    CUDA_CHECK(cudaGetLastError());
}

void return_expert_rows(const ExpertPeerBuffers* p, int peers, const Tensor& shared, Tensor& output,
                        MoeTokenValidity validity, MoeEpilogue mode, CudaExecutionView exec,
                        const Tensor* shared_gate) {
    if (!p || peers <= 0 || (mode != MoeEpilogue::Write && mode != MoeEpilogue::AddResidual))
        throw std::invalid_argument("invalid expert return operands");
    const int h = output.ne[0], n = output.ne[1];
    common::operand::tensor(shared, DType::FP32, {h, n}, 4, "expert return", "shared",
                            exec.device());
    common::operand::tensor(output, DType::BF16, {h, n}, 2, "expert return", "output",
                            exec.device());
    validity_view(validity, n, exec.device());
    if (shared_gate)
        common::operand::tensor(*shared_gate, DType::FP32, {n}, alignof(float), "expert return",
                                "shared gate", exec.device());
    combine<<<dim3((output.ne[0] + 255) / 256, output.ne[1]), 256, 0, exec.stream()>>>(
        p, peers, static_cast<const float*>(shared.data), static_cast<__nv_bfloat16*>(output.data),
        output.ne[0], static_cast<const int*>(validity.valid_columns.data), validity.tokens_per_row,
        mode == MoeEpilogue::AddResidual,
        shared_gate ? static_cast<const float*>(shared_gate->data) : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

void observe_expert_packets(const ExpertPeerBuffers* p, int peers, Tensor& totals,
                            CudaExecutionView exec) {
    if (!p || peers <= 0) throw std::invalid_argument("invalid expert observation peers");
    common::operand::tensor(totals, DType::U64, {4, peers}, 8, "expert observation", "totals",
                            exec.device());
    observe<<<(peers + 127) / 128, 128, 0, exec.stream()>>>(
        p, peers, static_cast<std::uint64_t*>(totals.data));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
