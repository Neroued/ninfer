#pragma once

#include "ops/quantized_weight.cuh"
#include "../../../../tools/support/kpool_chain_fixture.h"
#include "ninfer/ops/linear.h"

namespace ninfer::bench::kpool_application {
namespace kp = ninfer::tool::kpool;

// Generate represented histories outside timing, without a multi-GiB Host mirror.
// These are synthetic finite cache operands, not a claim about model activations.
static __global__ void history_words(std::byte* base, std::size_t stride, std::size_t offset,
                                     int words, int pages, int multiplier, unsigned seed) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < std::size_t(words) * pages; i += std::size_t(gridDim.x) * blockDim.x) {
        const float value = fixture::uniform(i, seed, -1.F, 1.F);
        const auto physical =
            multiplier ? ((i / words) * multiplier + pages / 2) % pages : i / words;
        auto* page      = reinterpret_cast<__nv_bfloat16*>(base + physical * stride + offset);
        page[i % words] = __float2bfloat16_rn(value);
    }
}

enum class Phase { Full, Reuse, History };

struct Fixture : kp::ChainFixture {
    PackedQuantizedWeight o_weight;
    kp::Buffer<std::uint16_t> result, raw_keys, raw_gates;
    kp::Buffer<int> record_offsets, bases, accepted;
    bool record;
    std::size_t o_capacity;

    Fixture(CudaExecutionView cuda, int T, int B, const kp::IndexerCase& shape, bool record_mode,
            int layers = 11)
        : ChainFixture(cuda, T, B, *std::max_element(shape.ends.begin(), shape.ends.end()), layers,
                       false, shape, false, false),
          o_weight(make_row_split_weight(QType::Q8_G32_FP16, 4096, 16384, 16384, 20261024)),
          result(cuda.device(), DType::BF16, {4096, T, B}),
          raw_keys(cuda.device(), DType::BF16, {128, T, B}),
          raw_gates(cuda.device(), DType::BF16, {128, T, B}),
          record_offsets(cuda.device(), DType::I32, {B}), bases(cuda.device(), DType::I32, {B}),
          accepted(cuda.device(), DType::I32, {B}), record(record_mode),
          o_capacity(ops::linear_workspace_capacity_bytes(QType::Q8_G32_FP16, 4096, 16384,
                                                          ops::LinearPolicy::A16Only, B, T * B)) {
        auto& i        = index;
        int multiplier = 17;
        while (std::gcd(multiplier, i.physical_pages) != 1) multiplier += 2;
        for (int plane : {2 * i.layer, 2 * i.layer + 1}) {
            const auto& p = i.packed.layout().planes[plane];
            history_words<<<4096, 256, 0, cuda.stream()>>>(
                static_cast<std::byte*>(i.packed.storage().data), i.packed.record_stride(),
                i.packed.physical_offset(0) + p.tensor.region.offset,
                int(p.tensor.region.bytes / 2), i.physical_pages,
                shape.contiguous_pages ? 0 : multiplier, 20261024 + plane);
            CUDA_CHECK(cudaGetLastError());
        }
        for (int b = 0; b < B; ++b) {
            record_offsets.at(b) = 0;
            bases.at(b)          = i.valid.at(b) ? i.positions.at(0, b) : 0;
            accepted.at(b)       = i.valid.at(b);
        }
        record_offsets.upload();
        bases.upload();
        accepted.upload();
        reserve(cuda, {int(index.envelope.max_visible_tokens)});
        CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    }

    static Weight prefix(Weight w, int rows) {
        w.n = w.shape[0] = w.padded_shape[0] = rows;
        w.payload_bytes                      = std::size_t(rows) * w.k * 2;
        return w;
    }

    ops::KpoolMlaPrepareParameters params(Phase phase) const {
        auto p = weights.params;
        if (phase == Phase::Full) return p;
        p.projection =
            phase == Phase::History ? kp::Projection::History : kp::Projection::QueryAndHistory;
        p.a_projection = prefix(p.a_projection, phase == Phase::History ? 768 : 2304);
        if (phase == Phase::History) p.query_norm = {};
        return p;
    }

    // Reserve once for every captured envelope and every measured phase. No allocation
    // or weight materialization belongs to the captured bodies below.
    void reserve(CudaExecutionView cuda, const std::vector<int>& bounds) {
        auto cap              = std::max(o_capacity, capacities[1]);
        auto prepare_capacity = record ? ops::kpool_mla_prepare_record_workspace_capacity_bytes
                                       : ops::kpool_mla_prepare_update_workspace_capacity_bytes;
        for (auto phase : {Phase::Full, Phase::Reuse, Phase::History}) {
            const auto p = params(phase);
            cap          = std::max(cap, prepare_capacity({p.projection, 4096, 512, 128,
                                                  phase == Phase::History ? 0 : 1536,
                                                  phase == Phase::Full ? 32 : 0, 4},
                                                          ops::KpoolMlaStorage::BFloat16, index.B, 1,
                                                          index.T, cuda.multiprocessor_count()));
        }
        cap = std::max(cap, ops::linear_workspace_capacity_bytes(QType::BF16, 16384, 1536,
                                                                 ops::LinearPolicy::A16Only,
                                                                 index.B, index.T * index.B));
        for (int u : bounds) {
            cap = std::max(cap, ops::kpool_indexer_workspace_capacity_bytes(
                                    {32, 128}, {std::uint32_t(u)}, index.B, 1, index.T,
                                    cuda.multiprocessor_count()));
            cap = std::max(cap, ops::kpool_sparse_mla_workspace_capacity_bytes(
                                    {64, 256, 256, 512}, ops::KpoolMlaStorage::BFloat16,
                                    {std::uint32_t(u)}, index.B, 1, index.T,
                                    cuda.multiprocessor_count()));
        }
        index.workspace.reset();
        index.scratch = DeviceBuffer(cuda.device(), cap + 512);
        index.scratch.fill(0xa5);
        index.workspace_bytes = cap;
        index.workspace = std::make_unique<MemoryArena>(index.scratch.span().subspan(256, cap));
    }

    void prepare_phase(CudaExecutionView cuda, Phase phase = Phase::Full) {
        const auto p = params(phase);
        const ops::KpoolMlaPrepareOutputs outputs{phase == Phase::History ? Tensor{} : rank.view,
                                                  phase == Phase::Full ? index.coeff.view
                                                                       : Tensor{}};
        const ops::KpoolMlaBatchView batch{index.positions.view, index.valid.view,
                                           index.selectors.view};
        if (record)
            ops::kpool_mla_prepare_record(x.view, p, batch, before, cache,
                                          {raw_keys.view, raw_gates.view, record_offsets.view},
                                          outputs, *index.workspace, cuda);
        else
            ops::kpool_mla_prepare_update(x.view, p, batch, before, cache, after, outputs,
                                          *index.workspace, cuda);
    }

    void queries_phase(CudaExecutionView cuda, Phase phase) {
        if (phase == Phase::Full) {
            queries(cuda);
            return;
        }
        auto out = query.view.reshape({16384, index.T * index.B});
        ops::linear(rank.view.reshape({1536, index.T * index.B}), prefix(query_weight, 16384), out,
                    ops::LinearPolicy::A16Only, *index.workspace, cuda);
    }

    void project_o(CudaExecutionView cuda) {
        auto out = result.view.reshape({4096, index.T * index.B});
        ops::linear(output.view.reshape({16384, index.T * index.B}), o_weight.weight, out,
                    ops::LinearPolicy::A16Only, *index.workspace, cuda);
    }

    void launch(CudaExecutionView cuda, Phase phase = Phase::Full) {
        prepare_phase(cuda, phase);
        if (phase == Phase::History) return;
        queries_phase(cuda, phase);
        if (phase == Phase::Full) index.launch(cuda);
        attend(cuda);
        project_o(cuda);
    }

    void extract(CudaExecutionView cuda) {
        ops::kpool_tail_extract(before,
                                {raw_keys.view.reshape({128, index.T, index.B, 1}),
                                 raw_gates.view.reshape({128, index.T, index.B, 1}), bases.view,
                                 index.valid.view},
                                accepted.view, after, cuda);
    }

    void restore(CudaExecutionView cuda) {
        CUDA_CHECK(cudaMemcpyAsync(state.data(), initial_state.data(), state.size(),
                                   cudaMemcpyHostToDevice, cuda.stream()));
    }
};
} // namespace ninfer::bench::kpool_application
