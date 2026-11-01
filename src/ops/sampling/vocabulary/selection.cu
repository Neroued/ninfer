#include "ninfer/ops/vocabulary.h"
#include "ops/common/host/operands.h"
#include "ops/sampling/kernels/distribution.cuh"

#include <algorithm>

namespace ninfer::ops {
namespace {
using namespace sampling_detail;
using namespace common;
using Schedule            = TopKSchedule<128, 4, 128, 4, 25>;
constexpr int kItems      = Schedule::kSamplerItemsPerThread;
constexpr int kGreedyTile = 1024;
constexpr int kColumnSlab = 65535;
constexpr const char* kOp = "vocabulary_candidates";

struct Plan {
    int partials;
    std::size_t first_bytes, second_bytes;

    std::size_t bytes() const { return first_bytes + second_bytes; }
};

Plan plan(int rows, std::int64_t columns, int cap) {
    if (rows <= 0 || columns <= 0 || (cap != 1 && cap != 20))
        throw std::invalid_argument("vocabulary_candidates: invalid workspace profile");
    columns            = std::min(columns, std::int64_t(kColumnSlab));
    const int partials = (rows - 1) / (cap == 1 ? kGreedyTile : 512) + 1;
    const auto bytes   = [&](int lists) {
        return (std::size_t(lists) * columns * cap * 8 + 255) & ~std::size_t(255);
    };
    return {partials, partials == 1 ? 0 : bytes(partials),
            cap == 20 && partials > 25 ? bytes(div_up(partials, 25)) : 0};
}

__device__ float adjusted(float raw, int id, const SamplingConfig& cfg, TokenRow drafts,
                          int column) {
    // Mask positions are independent of the optional penalty overlay.
    if (cfg.mask.words &&
        !(cfg.mask.words[std::int64_t(column) * cfg.mask.stride + id / 32] & (1U << (id % 32))))
        return -CUDART_INF_F;
    auto unmasked = cfg;
    unmasked.mask = {};
    return sampling_adjusted_logit(raw, id, unmasked, drafts, drafts ? column : 0);
}

template <bool Greedy>
__global__ void partial_kernel(const __nv_bfloat16* logits, int rows, int begin, int domain,
                               const SamplingConfig* configs, const int* valid, TokenRows drafts,
                               int tokens, std::int64_t column_offset, unsigned long long* output,
                               int partials) {
    const int column         = blockIdx.y;
    const auto global_column = column_offset + column;
    const int b = global_column / tokens, t = global_column % tokens;
    const int start    = blockIdx.x * (Greedy ? kGreedyTile : 512);
    const bool active  = t < valid[b];
    const auto cfg     = configs[b];
    const auto overlay = drafts[b];
    const auto base    = global_column * rows;
    if constexpr (Greedy) {
        unsigned long long key = 0;
        for (int offset = threadIdx.x; offset < min(kGreedyTile, rows - start);
             offset += blockDim.x) {
            const int local = start + offset;
            if (active && std::int64_t(begin) + local < domain) {
                const int id = begin + local;
                const float value =
                    adjusted(__bfloat162float(logits[base + local]), id, cfg, overlay, t);
                if (isfinite(value))
                    key = max(key, static_cast<unsigned long long>(score_id_order_key(value, id)));
            }
        }
        __shared__ unsigned long long warp_keys[4];
        key = block_reduce_max_key(key, warp_keys);
        if (threadIdx.x == 0) output[std::int64_t(column) * partials + blockIdx.x] = key;
    } else {
        __shared__ SamplingPartialTopKStorage<Schedule> storage;
        Workspace workspace{};
        workspace.partial_keys   = output + std::int64_t(column) * partials * 20;
        workspace.partial_stride = partials;
        const int cap            = cfg.temperature > 0 ? sampling_candidate_cap(cfg, domain) : 1;
        if (cfg.presence_penalty == 0 && cfg.frequency_penalty == 0) {
            unsigned int keys[kItems];
#pragma unroll
            for (int i = 0; i < kItems; ++i) {
                const int index = threadIdx.x * kItems + i;
                keys[i]         = 0;
                if (active && index < rows - start) {
                    const int local = start + index;
                    if (std::int64_t(begin) + local >= domain) continue;
                    const int id = begin + local;
                    if (!cfg.mask.words ||
                        (cfg.mask.words[std::int64_t(t) * cfg.mask.stride + id / 32] &
                         (1U << (id % 32))))
                        keys[i] =
                            sampling_bf16_tile_sort_key<Schedule>(logits[base + local], index);
                }
            }
            const int first_id = std::int64_t(begin) + start < domain ? begin + start : 0;
            sampling_store_bf16_tile_topk<Schedule>(keys, cap, first_id, workspace, 0, blockIdx.x,
                                                    storage.bf16);
        } else {
            unsigned long long keys[kItems];
#pragma unroll
            for (int i = 0; i < kItems; ++i) {
                const int index = threadIdx.x * kItems + i;
                keys[i]         = 0;
                if (active && index < rows - start) {
                    const int local = start + index;
                    if (std::int64_t(begin) + local >= domain) continue;
                    const int id = begin + local;
                    const float value =
                        adjusted(__bfloat162float(logits[base + local]), id, cfg, overlay, t);
                    if (isfinite(value)) keys[i] = score_id_order_key(value, id);
                }
            }
            sampling_store_tile_topk<Schedule>(keys, cap, workspace, 0, blockIdx.x, storage.fp32);
        }
        // The helper only writes cap entries; every physical key has a defined value.
        if (threadIdx.x < 20 && threadIdx.x >= cap)
            output[(std::int64_t(column) * partials + blockIdx.x) * 20 + threadIdx.x] = 0;
    }
}

__global__ void merge_greedy(const unsigned long long* input, unsigned long long* output,
                             int partials) {
    unsigned long long key = 0;
    for (int i = threadIdx.x; i < partials; i += blockDim.x)
        key = max(key, input[std::int64_t(blockIdx.x) * partials + i]);
    __shared__ unsigned long long warp_keys[4];
    key = block_reduce_max_key(key, warp_keys);
    if (threadIdx.x == 0) output[blockIdx.x] = key;
}

__global__ void merge_topk(const unsigned long long* input, unsigned long long* output, int lists,
                           int groups, const SamplingConfig* configs, int tokens,
                           std::int64_t column_offset) {
    const int begin = blockIdx.x * 25 * 20;
    const auto cfg  = configs[(column_offset + blockIdx.y) / tokens];
    const int cap   = cfg.temperature > 0 ? sampling_candidate_cap(cfg, 20) : 1;
    unsigned long long keys[kItems];
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
        const int index = threadIdx.x * kItems + i;
        keys[i]         = index < 500 && begin + index < lists * 20
                              ? input[std::int64_t(blockIdx.y) * lists * 20 + begin + index]
                              : 0;
    }
    __shared__ typename SamplingGroupSort<Schedule>::TempStorage storage;
    SamplingGroupSort<Schedule>(storage).Sort(keys, ScoreIdOrderGreater{});
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
        const int index = threadIdx.x * kItems + i;
        if (index < 20)
            output[(std::int64_t(blockIdx.y) * groups + blockIdx.x) * 20 + index] =
                index < cap ? keys[i] : 0;
    }
}
} // namespace

std::size_t vocabulary_candidates_workspace_capacity_bytes(int physical_rows,
                                                           std::int64_t max_columns,
                                                           int candidates) {
    return plan(physical_rows, max_columns, candidates).bytes();
}

void vocabulary_candidates(const Tensor& logits, int token_begin, int token_domain,
                           const SamplingConfig* configs, const Tensor& valid_columns,
                           const Tensor& drafts, Tensor& keys, MemoryArena& workspace,
                           CudaExecutionView execution) {
    const int rows = logits.ne[0], tokens = logits.ne[1], batch = logits.ne[2], cap = keys.ne[0];
    if (tokens <= 0 || batch < 1 || batch > 8 || token_begin < 0 || token_domain <= 0)
        throw std::invalid_argument("vocabulary_candidates: invalid vocabulary or T/B");
    const auto columns  = std::int64_t(tokens) * batch;
    const auto selected = plan(rows, columns, cap);
    operand::tensor(logits, DType::BF16, {rows, tokens, batch}, 2, kOp, "logits",
                    execution.device());
    operand::tensor(keys, DType::I64, {cap, tokens, batch}, 8, kOp, "keys", execution.device());
    operand::tensor(valid_columns, DType::I32, {batch}, 4, kOp, "valid columns",
                    execution.device());
    if (drafts.data) {
        if (tokens < 2) throw std::invalid_argument("vocabulary_candidates: drafts require T>=2");
        operand::strided_tensor(drafts, DType::I32, {tokens - 1, batch}, 4, kOp, "drafts",
                                execution.device());
    }
    if (!configs || !operand::aligned(configs, alignof(SamplingConfig)))
        throw std::invalid_argument("vocabulary_candidates: invalid configs");
    const auto destination = operand::range(keys, kOp);
    for (const auto* input : {&logits, &valid_columns, &drafts})
        if (input->data) operand::disjoint(destination, operand::strided_range(*input, kOp), kOp);
    const auto config_range = operand::range(configs, sizeof(SamplingConfig) * batch, kOp);
    operand::disjoint(destination, config_range, kOp);
    operand::device(workspace.device(), execution.device(), kOp, "workspace");
    auto scope                = workspace.scope();
    auto* output              = static_cast<unsigned long long*>(keys.data);
    unsigned long long *first = output, *second = nullptr;
    if (selected.bytes()) {
        const auto scratch = workspace.alloc_bytes(selected.bytes());
        const auto range   = operand::range(scratch.data, scratch.bytes, kOp);
        operand::disjoint(range, destination, kOp);
        operand::disjoint(range, config_range, kOp);
        for (const auto* input : {&logits, &valid_columns, &drafts})
            if (input->data) operand::disjoint(range, operand::strided_range(*input, kOp), kOp);
        first  = static_cast<unsigned long long*>(scratch.data);
        second = reinterpret_cast<unsigned long long*>(static_cast<char*>(scratch.data) +
                                                       selected.first_bytes);
    }
    for (std::int64_t offset = 0; offset < columns; offset += kColumnSlab) {
        const int width = int(std::min(columns - offset, std::int64_t(kColumnSlab)));
        auto* result    = output + offset * cap;
        auto* partial   = selected.partials == 1 ? result : first;
        const dim3 grid(selected.partials, width);
        if (cap == 1) {
            partial_kernel<true><<<grid, 128, 0, execution.stream()>>>(
                static_cast<const __nv_bfloat16*>(logits.data), rows, token_begin, token_domain,
                configs, static_cast<const int*>(valid_columns.data), token_rows(drafts), tokens,
                offset, partial, selected.partials);
            if (selected.partials > 1)
                merge_greedy<<<width, 128, 0, execution.stream()>>>(partial, result,
                                                                    selected.partials);
        } else {
            partial_kernel<false><<<grid, Schedule::kSamplerBlock, 0, execution.stream()>>>(
                static_cast<const __nv_bfloat16*>(logits.data), rows, token_begin, token_domain,
                configs, static_cast<const int*>(valid_columns.data), token_rows(drafts), tokens,
                offset, partial, selected.partials);
            auto* next = second;
            for (int lists = selected.partials; lists > 1;) {
                const int groups  = div_up(lists, 25);
                auto* destination = groups == 1 ? result : next;
                merge_topk<<<dim3(groups, width), Schedule::kSamplerGroupBlock, 0,
                             execution.stream()>>>(partial, destination, lists, groups, configs,
                                                   tokens, offset);
                std::swap(partial, next);
                lists = groups;
            }
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
