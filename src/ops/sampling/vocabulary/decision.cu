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
    const int rank    = threadIdx.x & 31;
    const auto* first = input + (std::int64_t(row) * tokens + col) * cap;
    const int limit   = cfg.temperature > 0 ? sampling_candidate_cap(cfg, 20) : 1;
    if (ranks <= 2) {
        auto key = rank < cap ? first[rank] : 0ULL;
        if (ranks == 2) {
            // Pad each sorted list to 32, reverse the second list, and take the
            // upper half of a 64-element bitonic merge. Only five shuffles remain.
            const int reverse = 31 - rank;
            const auto other =
                reverse < cap ? first[std::int64_t(batch) * tokens * cap + reverse] : 0ULL;
            key = key > other ? key : other;
#pragma unroll
            for (int distance = 16; distance; distance /= 2) {
                const auto value = __shfl_xor_sync(0xffffffffU, key, distance);
                key              = (rank & distance) == 0 ? (key > value ? key : value)
                                                          : (key < value ? key : value);
            }
        }
        const int count = __popc(__ballot_sync(0xffffffffU, rank < limit && key != 0));
        if (rank < count) {
            s.scores[rank] = score_from_order_key(key);
            s.ids[rank]    = id_from_order_key(key);
        }
        if (rank == 0) {
            s.size             = count;
            s.probabilities[0] = 1;
        }
    } else {
        const auto* list = rank < ranks
                               ? input + ((std::int64_t(rank) * batch + row) * tokens + col) * cap
                               : nullptr;
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
    }
    __syncwarp();
    if (rank == 0 && s.size && cfg.temperature > 0)
        sampling_normalize_support_values(cfg, s.scores, s.probabilities, &s.size, s.size);
    __syncwarp();
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
                              int ranks, const SamplingConfig* configs, TokenRows drafts,
                              const int* extents, const int* positions, int* licensed,
                              int* counts) {
    const int row = blockIdx.x, extent = extents[row], old_length = positions[row];
    const auto cfg = configs[row];
    __shared__ Support s;
    __shared__ int terminal;
    for (int col = 0; col <= extent; ++col) {
        distribution(keys, cap, drafts_count + 1, batch, ranks, row, col, cfg, s);
        if (threadIdx.x == 0) {
            if (!s.size) asm volatile("trap;");
            terminal        = -1;
            const int draft = col < extent ? drafts[row][col] : -1;
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
                speculative_store_accept_result(drafts[row], drafts_count, row, col, terminal,
                                                licensed + row * (drafts_count + 1), counts);
        }
        __syncthreads();
        if (terminal >= 0) break;
    }
}

__device__ float cumulative_weight(float weight) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        const float earlier = __shfl_up_sync(0xffffffffU, weight, offset);
        if (lane >= offset) weight += earlier;
    }
    return weight;
}

// Given a supplied path, acceptance events at each column are independent. One warp per
// column computes p and its event; a final warp chooses the first rejection. This bounds
// the critical path by a single Top-K merge instead of D serial distribution builds.
__global__ void accept_sparse_kernel(const unsigned long long* keys, int cap, int d, int batch,
                                     int ranks, const SamplingConfig* configs, TokenRows drafts,
                                     const int* ids, const float* q, const int* extents,
                                     const int* positions, int* licensed, int* counts) {
    const int row = blockIdx.x, col = threadIdx.x / 32, lane = threadIdx.x & 31;
    const int extent = extents[row], position = positions[row];
    const auto cfg = configs[row];
    __shared__ Support supports[16];
    __shared__ int rejected[16], terminal[16];
    if (col <= extent) {
        auto& s = supports[col];
        distribution(keys, cap, d + 1, batch, ranks, row, col, cfg, s);
        if (!s.size) asm volatile("trap;");
        const int draft = col < extent ? drafts[row][col] : -1;
        bool reject     = false;
        int selected    = s.ids[0];
        if (!(cfg.temperature > 0)) {
            reject = col < extent && draft != selected;
        } else {
            const int token = lane < s.size ? s.ids[lane] : INT_MAX;
            float weight    = lane < s.size ? s.probabilities[lane] : 0.0F;
            if (col < extent) {
                const int at      = (row * d + col) * 16 + lane;
                const float own_q = lane < 16 ? q[at] : 0.0F;
                const int own_id  = own_q > 0 ? ids[at] : INT_MAX;
                const float pd    = warp_sum(token == draft ? weight : 0.0F);
                const float qd    = warp_sum(own_id == draft ? own_q : 0.0F);
                const float u     = sampling_uniform(cfg.seed, position + col + 1,
                                                     kSamplePurposeSpeculativeAccept, 0);
                reject            = !(pd >= qd || u * qd < pd);
                if (reject) {
                    float qt = 0;
#pragma unroll
                    for (int c = 0; c < 16; ++c) {
                        const int id            = __shfl_sync(0xffffffffU, own_id, c);
                        const float probability = __shfl_sync(0xffffffffU, own_q, c);
                        if (token == id) qt = probability;
                    }
                    weight = fmaxf(weight - qt, 0.0F);
                }
            }
            if (reject || col == extent) {
                float cdf  = cumulative_weight(weight);
                float mass = __shfl_sync(0xffffffffU, cdf, 31);
                // Rounded p/q can differ by an ulp at every entry. A zero residual
                // then has no normalizable correction; use the target law.
                if (!(mass > 0)) {
                    weight = lane < s.size ? s.probabilities[lane] : 0.0F;
                    cdf    = cumulative_weight(weight);
                    mass   = __shfl_sync(0xffffffffU, cdf, 31);
                }
                const int purpose =
                    reject ? kSamplePurposeSpeculativeCorrection : kSamplePurposeSpeculativeBonus;
                const float goal =
                    sampling_uniform(cfg.seed, position + col + 1, purpose, 0) * mass;
                const unsigned positive = __ballot_sync(0xffffffffU, weight > 0);
                const unsigned hits     = __ballot_sync(0xffffffffU, weight > 0 && goal < cdf);
                const int pick          = hits ? __ffs(hits) - 1 : 31 - __clz(positive);
                selected                = __shfl_sync(0xffffffffU, token, pick);
            }
        }
        if (lane == 0) {
            rejected[col] = reject;
            terminal[col] = selected;
        }
    }
    __syncthreads();
    if (col == 0) {
        const unsigned failures = __ballot_sync(0xffffffffU, lane < extent && rejected[lane]);
        const int accepted      = failures ? __ffs(failures) - 1 : extent;
        speculative_sparse_warp_store(drafts, d, row, accepted, terminal[accepted], licensed,
                                      counts);
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
        for (const auto* input : inputs)
            operand::disjoint(range, operand::strided_range(*input, op), op);
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
                                  const Tensor& base_positions, Tensor& licensed_tokens,
                                  Tensor& licensed_counts, CudaExecutionView execution) {
    constexpr const char* op = "accept_vocabulary_candidates";
    check_keys(keys, configs, op, execution.device());
    const int d = keys.ne[1] - 1, batch = keys.ne[2];
    if (d < 1) throw std::invalid_argument("accept_vocabulary_candidates: expected D>=1");
    operand::strided_tensor(drafts, DType::I32, {d, batch}, 4, op, "drafts", execution.device());
    operand::tensor(current_extents, DType::I32, {batch}, 4, op, "extents", execution.device());
    operand::tensor(base_positions, DType::I32, {batch}, 4, op, "base positions",
                    execution.device());
    operand::tensor(licensed_counts, DType::I32, {batch}, 4, op, "counts", execution.device());
    operand::tensor(licensed_tokens, DType::I32, {d + 1, batch}, 4, op, "licensed tokens",
                    execution.device());
    check_outputs({&licensed_tokens, &licensed_counts},
                  {&keys, &drafts, &current_extents, &base_positions}, configs, batch, op);
    accept_kernel<<<batch, 32, 0, execution.stream()>>>(
        static_cast<const unsigned long long*>(keys.data), keys.ne[0], d, batch, keys.ne[3],
        configs, token_rows(drafts), static_cast<const int*>(current_extents.data),
        static_cast<const int*>(base_positions.data), static_cast<int*>(licensed_tokens.data),
        static_cast<int*>(licensed_counts.data));
    CUDA_CHECK(cudaGetLastError());
}

void accept_vocabulary_candidates_sparse(const Tensor& keys, const SamplingConfig* configs,
                                         const Tensor& drafts, const Tensor& candidate_ids,
                                         const Tensor& proposal_q, const Tensor& current_extents,
                                         const Tensor& base_positions, Tensor& licensed_tokens,
                                         Tensor& licensed_counts, CudaExecutionView execution) {
    constexpr const char* op = "accept_vocabulary_candidates_sparse";
    check_keys(keys, configs, op, execution.device());
    const int d = keys.ne[1] - 1, batch = keys.ne[2];
    if (d < 1) throw std::invalid_argument(std::string(op) + ": expected D>=1");
    operand::strided_tensor(drafts, DType::I32, {d, batch}, 4, op, "drafts", execution.device());
    operand::tensor(candidate_ids, DType::I32, {16, d, batch}, 4, op, "candidate IDs",
                    execution.device());
    operand::tensor(proposal_q, DType::FP32, {16, d, batch}, 4, op, "proposal q",
                    execution.device());
    operand::tensor(current_extents, DType::I32, {batch}, 4, op, "extents", execution.device());
    operand::tensor(base_positions, DType::I32, {batch}, 4, op, "positions", execution.device());
    operand::tensor(licensed_tokens, DType::I32, {d + 1, batch}, 4, op, "tokens",
                    execution.device());
    operand::tensor(licensed_counts, DType::I32, {batch}, 4, op, "counts", execution.device());
    check_outputs({&licensed_tokens, &licensed_counts},
                  {&keys, &drafts, &candidate_ids, &proposal_q, &current_extents, &base_positions},
                  configs, batch, op);
    accept_sparse_kernel<<<batch, (d + 1) * 32, 0, execution.stream()>>>(
        static_cast<const unsigned long long*>(keys.data), keys.ne[0], d, batch, keys.ne[3],
        configs, token_rows(drafts), static_cast<const int*>(candidate_ids.data),
        static_cast<const float*>(proposal_q.data), static_cast<const int*>(current_extents.data),
        static_cast<const int*>(base_positions.data), static_cast<int*>(licensed_tokens.data),
        static_cast<int*>(licensed_counts.data));
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
