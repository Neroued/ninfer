#include "ninfer/ops/vocabulary.h"
#include "ops/common/host/operands.h"
#include "ops/sampling/acceptance/kernel.cuh"

namespace ninfer::ops {
namespace {
using namespace sampling_detail;
using namespace common;

struct Support {
    float scores[20], probabilities[20];
    int ids[20], size;
};

__device__ void distribution(const unsigned long long* input, int cap, int tokens, int batch,
                             int ranks, int row, int col, const SamplingConfig& cfg, Support& s) {
    const int rank = threadIdx.x;
    const auto* list =
        rank < ranks ? input + ((std::int64_t(rank) * batch + row) * tokens + col) * cap : nullptr;
    const int limit = cfg.temperature > 0 ? sampling_candidate_cap(cfg, 20) : 1;
    int cursor = 0, count = 0;
    for (; count < limit; ++count) {
        const auto key       = list && cursor < cap ? list[cursor] : 0ULL;
        const auto high      = static_cast<unsigned int>(key >> 32);
        const auto best_high = __reduce_max_sync(0xffffffffU, high);
        const auto low       = high == best_high ? static_cast<unsigned int>(key) : 0U;
        const auto best_low  = __reduce_max_sync(0xffffffffU, low);
        const auto best      = (static_cast<unsigned long long>(best_high) << 32) | best_low;
        if (!best) break;
        const auto owners = __ballot_sync(0xffffffffU, key == best);
        if (rank == __ffs(static_cast<int>(owners)) - 1) ++cursor;
        if (rank == 0) {
            s.scores[count] = score_from_order_key(best);
            s.ids[count]    = id_from_order_key(best);
        }
    }
    if (rank == 0) {
        s.size             = count;
        s.probabilities[0] = 1;
    }
    __syncthreads();
    if (s.size && cfg.temperature > 0)
        sampling_normalize_support(cfg, s.scores, s.ids, s.probabilities, &s.size, s.size);
}

__global__ void sample_kernel(const unsigned long long* keys, int cap, int batch, int ranks,
                              const SamplingConfig* configs, const int* positions, int purpose,
                              int* output) {
    __shared__ Support s;
    const int row  = blockIdx.x;
    const auto cfg = configs[row];
    distribution(keys, cap, 1, batch, ranks, row, 0, cfg, s);
    if (threadIdx.x == 0) {
        int token = -1;
        if (s.size)
            token = cfg.temperature > 0
                        ? sampling_pick_from_support(
                              s.ids, s.probabilities, s.size, -1,
                              sampling_uniform(cfg.seed, positions[row], purpose, 0))
                        : s.ids[0];
        output[row] = token;
    }
}

__global__ void accept_kernel(const unsigned long long* keys, int cap, int drafts_count, int batch,
                              int ranks, const SamplingConfig* configs, const int* drafts,
                              const int* extents, int* lengths, int* anchors, int* licensed,
                              int* counts, int* accepted) {
    const int row = blockIdx.x, extent = extents[row], old_length = lengths[row];
    const auto cfg = configs[row];
    __shared__ Support s;
    __shared__ int terminal;
    for (int col = 0; col <= extent; ++col) {
        distribution(keys, cap, drafts_count + 1, batch, ranks, row, col, cfg, s);
        if (threadIdx.x == 0) {
            if (!s.size) asm volatile("trap;");
            terminal        = -1;
            const int draft = col < extent ? drafts[row * drafts_count + col] : -1;
            if (!(cfg.temperature > 0)) {
                if (col == extent || draft != s.ids[0]) terminal = s.ids[0];
            } else if (col < extent) {
                float probability = 0;
                for (int j = 0; j < s.size; ++j)
                    if (s.ids[j] == draft) probability = s.probabilities[j];
                const float u = sampling_uniform(cfg.seed, old_length + col + 1,
                                                 kSamplePurposeSpeculativeAccept, 0);
                if (!(u < probability))
                    terminal = sampling_pick_from_support(
                        s.ids, s.probabilities, s.size, draft,
                        sampling_uniform(cfg.seed, old_length + col + 1,
                                         kSamplePurposeSpeculativeCorrection, 0));
            } else {
                terminal =
                    sampling_pick_from_support(s.ids, s.probabilities, s.size, -1,
                                               sampling_uniform(cfg.seed, old_length + col + 1,
                                                                kSamplePurposeSpeculativeBonus, 0));
            }
            if (terminal >= 0)
                speculative_store_accept_result<false>(
                    drafts + row * drafts_count, drafts_count, row, col, terminal, lengths, anchors,
                    licensed + row * (drafts_count + 1), counts, accepted, configs + row);
        }
        __syncthreads();
        if (terminal >= 0) break;
    }
}

void check_keys(const Tensor& keys, const SamplingConfig* configs, const char* op, Device device) {
    if ((keys.ne[0] != 1 && keys.ne[0] != 20) || keys.ne[1] < 1 || keys.ne[1] > 16 ||
        keys.ne[2] < 1 || keys.ne[2] > 8 || keys.ne[3] < 1 || keys.ne[3] > 16 || !configs ||
        !operand::aligned(configs, alignof(SamplingConfig)))
        throw std::invalid_argument(std::string(op) + ": invalid candidates/configs");
    operand::tensor(keys, DType::I64, {keys.ne[0], keys.ne[1], keys.ne[2], keys.ne[3]}, 8, op,
                    "keys", device);
}

void check_outputs(std::initializer_list<Tensor*> outputs,
                   std::initializer_list<const Tensor*> inputs, const SamplingConfig* configs,
                   int batch, const char* op) {
    for (auto current = outputs.begin(); current != outputs.end(); ++current) {
        const auto range = operand::range(**current, op);
        for (const auto* input : inputs) operand::disjoint(range, operand::range(*input, op), op);
        operand::disjoint(range, operand::range(configs, sizeof(SamplingConfig) * batch, op), op);
        for (auto earlier = outputs.begin(); earlier != current; ++earlier)
            operand::disjoint(range, operand::range(**earlier, op), op);
    }
}
} // namespace

void sample_vocabulary_candidates(const Tensor& keys, const SamplingConfig* configs,
                                  const Tensor& logical_positions, int purpose, Tensor& output,
                                  CudaExecutionView execution) {
    constexpr const char* op = "sample_vocabulary_candidates";
    check_keys(keys, configs, op, execution.device());
    if (keys.ne[1] != 1) throw std::invalid_argument("sample_vocabulary_candidates: expected T=1");
    const int batch = keys.ne[2];
    operand::tensor(logical_positions, DType::I32, {batch}, 4, op, "positions", execution.device());
    operand::tensor(output, DType::I32, {batch}, 4, op, "output", execution.device());
    check_outputs({&output}, {&keys, &logical_positions}, configs, batch, op);
    sample_kernel<<<batch, 32, 0, execution.stream()>>>(
        static_cast<const unsigned long long*>(keys.data), keys.ne[0], batch, keys.ne[3], configs,
        static_cast<const int*>(logical_positions.data), purpose, static_cast<int*>(output.data));
    CUDA_CHECK(cudaGetLastError());
}

void accept_vocabulary_candidates(const Tensor& keys, const SamplingConfig* configs,
                                  const Tensor& drafts, const Tensor& current_extents,
                                  Tensor& round_lengths, Tensor& round_anchors,
                                  Tensor& licensed_tokens, Tensor& licensed_counts,
                                  Tensor& accepted_drafts, CudaExecutionView execution) {
    constexpr const char* op = "accept_vocabulary_candidates";
    check_keys(keys, configs, op, execution.device());
    const int d = keys.ne[1] - 1, batch = keys.ne[2];
    if (d < 1) throw std::invalid_argument("accept_vocabulary_candidates: expected D>=1");
    operand::tensor(drafts, DType::I32, {d, batch}, 4, op, "drafts", execution.device());
    operand::tensor(current_extents, DType::I32, {batch}, 4, op, "extents", execution.device());
    for (const auto* tensor : {&round_lengths, &round_anchors, &licensed_counts, &accepted_drafts})
        operand::tensor(*tensor, DType::I32, {batch}, 4, op, "round output", execution.device());
    operand::tensor(licensed_tokens, DType::I32, {d + 1, batch}, 4, op, "licensed tokens",
                    execution.device());
    check_outputs(
        {&round_lengths, &round_anchors, &licensed_tokens, &licensed_counts, &accepted_drafts},
        {&keys, &drafts, &current_extents}, configs, batch, op);
    accept_kernel<<<batch, 32, 0, execution.stream()>>>(
        static_cast<const unsigned long long*>(keys.data), keys.ne[0], d, batch, keys.ne[3],
        configs, static_cast<const int*>(drafts.data),
        static_cast<const int*>(current_extents.data), static_cast<int*>(round_lengths.data),
        static_cast<int*>(round_anchors.data), static_cast<int*>(licensed_tokens.data),
        static_cast<int*>(licensed_counts.data), static_cast<int*>(accepted_drafts.data));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
