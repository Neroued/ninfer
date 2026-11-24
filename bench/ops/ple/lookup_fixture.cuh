#pragma once

#include "options.h"
#include "core/nf4.h"
#include "ninfer/ops/ple.h"

#include <bit>
#include <bitset>
#include <optional>
#include <unordered_set>

namespace ninfer::bench::ple {
namespace lookup {
inline constexpr int kShardRows         = 2500000;
inline constexpr int kShardCount        = 128;
inline constexpr std::size_t kPageBytes = ((kShardRows + 44) / 45) * 4096ULL;

inline __host__ __device__ std::uint8_t encoded_byte(std::size_t index, int shard) {
    const int in_page = index % 4096, in_row = in_page % 90;
    const auto value = fixture::bits(index, 1009 + shard);
    return in_page < 4050 && in_row >= 80 ? value % 127 : value & 255;
}

__global__ void fill_pages(std::uint8_t* data, std::size_t bytes, int shard) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < bytes;
         i += std::size_t(gridDim.x) * blockDim.x)
        data[i] = encoded_byte(i, shard);
}

struct Table {
    std::vector<DeviceBuffer> pages, decode;
    std::vector<ops::Nf4EmbeddingShard> shards;
    std::array<bool, kShardCount> owned{};
    ops::NgramHashParameters hash;

    explicit Table(const std::string& placement) {
        hash.multipliers = {0xd6e8feb86659fd93ULL, 0xa5a3564e27f8862bULL, 0x9e3779b97f4a7c15ULL};
        for (int h = 0; h < 16; ++h) {
            hash.vocab_sizes[h] = 8 * kShardRows - h * 2 - 1;
            hash.offsets[h]     = h * 8 * kShardRows;
        }
        for (int s = 0; s < kShardCount; ++s) {
            owned[s] = placement == "all" || (placement == "lower" && s < 64) ||
                       (placement == "upper" && s >= 64) ||
                       (placement == "alternating" && s % 2 == 0);
            if (!owned[s]) continue;
            pages.push_back(make_buffer(kPageBytes));
            decode.push_back(make_buffer(4096));
            fill_pages<<<256, 256>>>(static_cast<std::uint8_t*>(pages.back().data()), kPageBytes,
                                     s);
            CUDA_CHECK(cudaGetLastError());
            const auto lut = make_nf4_bf16_lut(0.75F + s * 0.03125F);
            decode.back().copy_from_host(lut.words.data(), 4096);
            shards.push_back({pages.back().span(),
                              Tensor(decode.back().span(), DType::BF16, {16, 128}),
                              std::uint32_t(s * kShardRows), kShardRows});
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};
} // namespace lookup

struct LookupFixture {
    DeviceBuffer ids, valid, tail, state_offsets, output;
    Tensor input, counts_view, embedding;
    ops::NgramTokenTailView before;
    const lookup::Table& table;
    const std::optional<ops::RaggedSequenceView> seq;
    std::vector<int> counts;
    std::uint64_t bytes = 0, selected_rows = 0, unique_rows = 0, lookup_bytes = 0;

    LookupFixture(const Options& options, int tokens, const lookup::Table& weights,
                  std::optional<ops::RaggedSequenceView> ragged_view = std::nullopt)
        : table(weights), seq(ragged_view) {
        const auto device  = device_context().device();
        const auto rows    = seq ? seq->num_seqs() : options.batch;
        const auto offsets = ragged::make_state_offsets(rows, options.state_form, 8, 4);
        if (seq) {
            for (int s = 0; s < rows; ++s) counts.push_back(seq->segment_length(s));
        } else {
            counts      = extents(options.valid, tokens, rows);
            valid       = measurement::make_i32(counts);
            counts_view = Tensor(valid.span(), DType::I32, {rows});
        }
        std::vector<int> host(std::size_t(tokens) * options.batch);
        std::vector<int> history(offsets.source_bytes() / sizeof(int));
        for (std::size_t t = 0; t < host.size(); ++t) host[t] = fixture::bits(t, 721) % 248000;
        for (std::size_t r = 0; r < offsets.num_source_records; ++r) {
            history[2 * r]     = 11 + r % 248000;
            history[2 * r + 1] = 19 + r % 248000;
        }
        std::unordered_set<std::uint32_t> visited;
        std::array<std::bitset<2048>, lookup::kShardCount> used_lut;
        for (int b = 0; b < rows; ++b) {
            if (!counts[b]) continue;
            const auto start = seq ? std::size_t(seq->segment_begin(b)) : std::size_t(b) * tokens;
            const auto* input_ids = host.data() + start;
            const auto* tail_ids  = history.data() + offsets.source_offsets_host[b] / sizeof(int);
            bytes += std::uint64_t(counts[b]) * 4 + sizeof(std::int64_t);
            if (!seq) bytes += 8;
            for (int t = 0; t < counts[b]; ++t) {
                const auto previous = std::uint64_t(t ? input_ids[t - 1] : tail_ids[1]);
                const auto older    = std::uint64_t(previous == 7 ? 7
                                                    : t >= 2      ? input_ids[t - 2]
                                                                  : tail_ids[t == 1]);
                const auto pair     = std::uint64_t(input_ids[t]) * weights.hash.multipliers[0] ^
                                  previous * weights.hash.multipliers[1];
                for (int h = 0; h < 16; ++h) {
                    const auto word    = h < 8 ? pair : pair ^ older * weights.hash.multipliers[2];
                    const auto modulus = weights.hash.vocab_sizes[h];
                    auto index         = std::bit_cast<std::int64_t>(word) % std::int64_t(modulus);
                    if (index < 0) index += modulus;
                    const auto row  = std::uint32_t(index + weights.hash.offsets[h]);
                    const int shard = row / lookup::kShardRows;
                    if (!weights.owned[shard]) continue;
                    ++selected_rows;
                    if (!visited.insert(row).second) continue;
                    const auto local   = row % lookup::kShardRows;
                    const auto address = std::size_t(local / 45) * 4096 + (local % 45) * 90;
                    for (int pair_index = 0; pair_index < 80; ++pair_index) {
                        const auto code = lookup::encoded_byte(address + pair_index, shard);
                        const auto scale =
                            lookup::encoded_byte(address + 80 + pair_index / 8, shard);
                        used_lut[shard].set(scale * 16 + (code & 15));
                        used_lut[shard].set(scale * 16 + (code >> 4));
                    }
                }
            }
        }
        unique_rows  = visited.size();
        lookup_bytes = unique_rows * 90;
        if (seq) bytes += offsets.num_source_records * 8;
        ids           = ragged::upload(device, std::span<const int>(host));
        tail          = ragged::upload(device, std::span<const int>(history));
        state_offsets = ragged::upload_offsets(device, offsets.source_offsets_host);
        output        = make_buffer(host.size() * 5120);
        input         = seq ? Tensor(ids.span(), DType::I32, {tokens})
                            : Tensor(ids.span(), DType::I32, {tokens, rows});
        embedding     = seq ? Tensor(output.span(), DType::BF16, {2560, tokens})
                            : Tensor(output.span(), DType::BF16, {2560, tokens, rows});
        before        = {{{tail.span(), ragged::offset_tensor(state_offsets)},
                          {{{0, 8, 4}, DType::I32, {2, 1, 1, 1}}, 1, 8}}};
        bytes += output.size() + std::uint64_t(rows) * sizeof(int) + lookup_bytes;
        for (const auto& lut : used_lut) bytes += lut.count() * 2;
    }

    void launch(CudaExecutionView execution) {
        if (seq)
            ops::ngram_embedding_lookup(input, *seq, before, table.hash, 7, table.shards, embedding,
                                        execution);
        else
            ops::ngram_embedding_lookup(input, counts_view, before, table.hash, 7, table.shards,
                                        embedding, execution);
    }

    void restore(CudaExecutionView) {}

    std::uint64_t logical_bytes() const { return bytes; }
};
} // namespace ninfer::bench::ple
