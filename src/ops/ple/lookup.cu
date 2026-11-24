#include "ninfer/ops/ple.h"
#include "ops/common/host/operands.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {
namespace operand         = common::operand;
constexpr const char* kOp = "ngram_embedding_lookup";
constexpr int kHeads = 16, kWidth = 160, kRowsPerPage = 45, kRowBytes = 90;

struct Shard {
    const std::uint8_t* pages;
    const std::uint16_t* decode;
    std::uint32_t begin, rows;
};

struct HashHead {
    std::uint64_t reciprocal;
    std::uint32_t vocab, offset;
    int first, last;
};

struct Arguments {
    const int *ids, *valid, *tail;
    const std::int64_t* state_offsets;
    std::uint32_t* out;
    std::int64_t id_stride;
    int tokens, batch, eos;
    std::uint64_t multipliers[3];
    HashHead heads[kHeads];
    Shard shards[128];
    const std::int32_t *cu_seqlens = nullptr, *token_to_seq = nullptr;
};

static_assert(sizeof(Arguments) <= 4096);

void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(std::string(kOp) + ": " + message);
}

Arguments arguments(const Tensor& ids, const Tensor& valid, const NgramTokenTailView& before,
                    const NgramHashParameters& hash, int eos,
                    std::span<const Nf4EmbeddingShard> shards, Tensor& out,
                    CudaExecutionView execution, const RaggedSequenceView* ragged = nullptr) {
    const int tokens = ids.ne[0], batch = ids.ne[1];
    const int rows    = ragged ? ragged->num_seqs() : batch;
    const auto device = execution.device();
    require(tokens > 0 && batch >= 1, "invalid T/B");
    require(eos >= 0 && shards.size() <= 128, "invalid EOS or shard count");
    operand::slotted_tensor(ids, DType::I32, {tokens, batch}, 1, 4, kOp, "token IDs", device);
    require(before.token_ids.field.layers == 1, "tail layer");
    operand::state_field(before.token_ids, DType::I32, {2}, 4, kOp, "tail IDs", device);
    if (!ragged) operand::tensor(valid, DType::I32, {batch}, 4, kOp, "valid columns", device);
    operand::shape(before.token_ids.records.byte_offsets, {rows, 1, 1, 1}, kOp,
                   "tail byte offsets");
    operand::tensor(out, DType::BF16, {kHeads * kWidth, tokens, batch}, 16, kOp, "embedding",
                    device);
    const auto destination = operand::range(out, kOp);
    for (const auto* input : {&ids, &valid, &before.token_ids.records.byte_offsets})
        require(!operand::overlaps(operand::blocks(*input, kOp), destination), "output alias");

    Arguments a{};
    a.ids           = static_cast<const int*>(ids.data);
    a.valid         = static_cast<const int*>(valid.data);
    a.tail          = operand::state_field_base<const int>(before.token_ids);
    a.state_offsets = static_cast<const std::int64_t*>(before.token_ids.records.byte_offsets.data);
    a.out           = static_cast<std::uint32_t*>(out.data);
    a.id_stride     = ids.nb[1] / 4;
    a.tokens        = tokens;
    a.batch         = batch;
    a.eos           = eos;
    if (ragged) {
        a.cu_seqlens   = static_cast<const std::int32_t*>(ragged->cu_seqlens.data);
        a.token_to_seq = static_cast<const std::int32_t*>(ragged->token_to_seq.data);
    }
    std::copy(hash.multipliers.begin(), hash.multipliers.end(), a.multipliers);
    std::uint64_t previous_end = 0;
    for (std::size_t i = 0; i < shards.size(); ++i) {
        const auto& shard = shards[i];
        const auto end    = std::uint64_t(shard.row_begin) + shard.rows;
        require(shard.rows && shard.row_begin >= previous_end && end <= (1ULL << 32),
                "overlapping or invalid shard rows");
        previous_end = end;
        operand::device(shard.pages.device, device, kOp, "row pages");
        const auto bytes = ((std::uint64_t(shard.rows) + kRowsPerPage - 1) / kRowsPerPage) * 4096;
        require(shard.pages.data && shard.pages.bytes >= bytes, "truncated row pages");
        operand::tensor(shard.decode, DType::BF16, {16, 128}, 2, kOp, "decode table", device);
        operand::disjoint(destination, operand::range(shard.pages.data, shard.pages.bytes, kOp),
                          kOp);
        operand::disjoint(destination, operand::range(shard.decode, kOp), kOp);
        a.shards[i] = {static_cast<const std::uint8_t*>(shard.pages.data),
                       static_cast<const std::uint16_t*>(shard.decode.data), shard.row_begin,
                       shard.rows};
    }
    for (int h = 0; h < kHeads; ++h) {
        const auto vocab = hash.vocab_sizes[h], offset = hash.offsets[h];
        const auto end = std::uint64_t(offset) + vocab;
        require(vocab && end <= (1ULL << 32), "invalid hash row domain");
        auto& head  = a.heads[h];
        head.vocab  = vocab;
        head.offset = offset;
        // floor(2^64/vocab), with divisor one handled separately by the kernel.
        const auto maximum = std::numeric_limits<std::uint64_t>::max();
        head.reciprocal    = vocab == 1 ? 0 : maximum / vocab + (maximum % vocab == vocab - 1);
        head.first         = int(shards.size());
        head.last          = head.first;
        for (int i = 0; i < int(shards.size()); ++i) {
            const auto& shard = a.shards[i];
            if (std::uint64_t(shard.begin) + shard.rows > offset && shard.begin < end) {
                head.first = std::min(head.first, i);
                head.last  = i + 1;
            }
        }
    }
    return a;
}

__device__ std::uint32_t hash_row(std::uint64_t word, const HashHead& head) {
    const bool negative     = (word >> 63) != 0;
    const auto magnitude    = negative ? 0ULL - word : word;
    std::uint64_t remainder = 0;
    if (head.vocab != 1) {
        const auto quotient = __umul64hi(magnitude, head.reciprocal);
        remainder           = magnitude - quotient * head.vocab;
        if (remainder >= head.vocab) remainder -= head.vocab;
        if (negative && remainder) remainder = head.vocab - remainder;
    }
    return head.offset + static_cast<std::uint32_t>(remainder);
}

template <int Warps, bool Ragged>
__global__ __launch_bounds__(Warps * 32) void lookup_kernel(const __grid_constant__ Arguments a) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const auto tasks = std::int64_t(a.tokens) * a.batch * kHeads;
    for (std::int64_t task = std::int64_t(blockIdx.x) * Warps + warp; task < tasks;
         task += std::int64_t(gridDim.x) * Warps) {
        const int head_index = task % kHeads;
        const auto column    = task / kHeads;
        int b = 0, t = 0;
        if constexpr (!Ragged) {
            b = column / a.tokens;
            t = column % a.tokens;
        }
        const auto& head  = a.heads[head_index];
        int owner         = -1;
        std::uint32_t row = 0;
        if (lane == 0 && head.first < head.last && (Ragged || t < a.valid[b])) {
            const int* ids;
            if constexpr (Ragged) {
                b   = a.token_to_seq[column];
                t   = column - a.cu_seqlens[b];
                ids = a.ids + column - t;
            } else {
                ids = a.ids + b * a.id_stride;
            }
            const auto offset   = t < 2 ? a.state_offsets[b] : 0;
            const auto* tail    = a.tail + offset / 4;
            const auto current  = std::uint64_t(ids[t]);
            const auto previous = std::uint64_t(t ? ids[t - 1] : tail[1]);
            const auto older    = std::uint64_t(previous == std::uint64_t(a.eos) ? a.eos
                                                : t >= 2                         ? ids[t - 2]
                                                : t == 1                         ? tail[1]
                                                                                 : tail[0]);
            auto word           = current * a.multipliers[0] ^ previous * a.multipliers[1];
            if (head_index >= 8) word ^= older * a.multipliers[2];
            row       = hash_row(word, head);
            int first = head.first, last = head.last;
            while (first < last) {
                const int middle = (first + last) / 2;
                if (a.shards[middle].begin <= row)
                    first = middle + 1;
                else
                    last = middle;
            }
            const int candidate = first - 1;
            if (candidate >= head.first &&
                row - a.shards[candidate].begin < a.shards[candidate].rows)
                owner = candidate;
        }
        owner                       = __shfl_sync(0xffffffffU, owner, 0);
        row                         = __shfl_sync(0xffffffffU, row, 0);
        const std::uint8_t* encoded = nullptr;
        const std::uint16_t* decode = nullptr;
        if (owner >= 0) {
            const auto& shard = a.shards[owner];
            const auto local  = row - shard.begin;
            encoded           = shard.pages + std::uint64_t(local / kRowsPerPage) * 4096 +
                      (local % kRowsPerPage) * kRowBytes;
            decode = shard.decode;
        }
        for (int pair = lane; pair < kWidth / 2; pair += 32) {
            std::uint32_t value = 0;
            if (owner >= 0) {
                const auto codes = encoded[pair], scale = encoded[80 + pair / 8];
                value = decode[scale * 16 + (codes & 15)] |
                        (std::uint32_t(decode[scale * 16 + (codes >> 4)]) << 16);
            }
            a.out[task * (kWidth / 2) + pair] = value;
        }
    }
}

template <bool Ragged>
void launch_lookup(const Arguments& a, CudaExecutionView execution) {
    const auto columns = std::int64_t(a.tokens) * a.batch;
    const auto tasks   = columns * kHeads;
    if (columns <= 8) {
        lookup_kernel<1, Ragged><<<int(tasks), 32, 0, execution.stream()>>>(a);
    } else {
        const int blocks =
            int(std::min((tasks + 7) / 8, std::int64_t(execution.multiprocessor_count()) * 16));
        lookup_kernel<8, Ragged><<<blocks, 256, 0, execution.stream()>>>(a);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void ngram_embedding_lookup(const Tensor& token_ids, const Tensor& valid_columns,
                            const NgramTokenTailView& before, const NgramHashParameters& hash,
                            std::int32_t eos, std::span<const Nf4EmbeddingShard> shards,
                            Tensor& embedding, CudaExecutionView execution) {
    launch_lookup<false>(
        arguments(token_ids, valid_columns, before, hash, eos, shards, embedding, execution),
        execution);
}

void ngram_embedding_lookup(const Tensor& token_ids, const RaggedSequenceView& ragged,
                            const NgramTokenTailView& before, const NgramHashParameters& hash,
                            std::int32_t eos, std::span<const Nf4EmbeddingShard> shards,
                            Tensor& embedding, CudaExecutionView execution) {
    const auto device = execution.device();
    operand::ragged_sequence(ragged, kOp, device);
    const int n = ragged.total_tokens(), rows = ragged.num_seqs();
    operand::tensor(token_ids, DType::I32, {n}, 4, kOp, "token IDs", device);
    operand::tensor(embedding, DType::BF16, {kHeads * kWidth, n}, 16, kOp, "embedding", device);
    const auto destination = operand::range(embedding, kOp);
    for (const auto* input : {&ragged.cu_seqlens, &ragged.seq_lengths, &ragged.token_to_seq})
        operand::disjoint(destination, operand::range(*input, kOp), kOp);
    const auto ids = token_ids.view({n, 1});
    if (rows == 1) {
        auto out = embedding.view({kHeads * kWidth, n, 1});
        ngram_embedding_lookup(ids, ragged.seq_lengths, before, hash, eos, shards, out, execution);
        return;
    }
    launch_lookup<true>(arguments(ids, ragged.seq_lengths, before, hash, eos, shards, embedding,
                                  execution, &ragged),
                        execution);
}

} // namespace ninfer::ops
