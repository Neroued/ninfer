#pragma once

#include "options.h"
#include "core/nf4.h"
#include "ninfer/ops/ple.h"

#include <bit>
#include <bitset>
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
    std::vector<int> counts;
    std::uint64_t bytes = 0, selected_rows = 0, unique_rows = 0, lookup_bytes = 0;

    LookupFixture(const Options& options, int tokens, const lookup::Table& weights)
        : ids(make_buffer(std::size_t(tokens) * options.batch * 4)),
          valid(make_buffer(options.batch * 4)), tail(make_buffer(options.batch * 2 * 4)),
          state_offsets(make_buffer(options.batch * 8)),
          output(make_buffer(std::size_t(tokens) * options.batch * 5120)),
          input(ids.span(), DType::I32, {tokens, options.batch}),
          counts_view(valid.span(), DType::I32, {options.batch}),
          embedding(output.span(), DType::BF16, {2560, tokens, options.batch}),
          before{{{tail.span(), Tensor(state_offsets.span(), DType::I64, {options.batch})},
                  {{{0, 8, 4}, DType::I32, {2, 1, 1, 1}}, 1, 8}}},
          table(weights), counts(extents(options.valid, tokens, options.batch)) {
        std::vector<int> host(std::size_t(tokens) * options.batch), history(options.batch * 2);
        std::vector<std::int64_t> select(options.batch);
        std::unordered_set<std::uint32_t> visited;
        std::array<std::bitset<2048>, lookup::kShardCount> used_lut;
        for (int b = 0; b < options.batch; ++b) {
            select[b]          = b * 8;
            history[2 * b]     = 11 + b;
            history[2 * b + 1] = 19 + b;
            for (int t = 0; t < tokens; ++t)
                host[b * tokens + t] = fixture::bits(b * tokens + t, 721) % 248000;
            if (!counts[b]) continue;
            bytes += std::uint64_t(counts[b]) * 4 + 16;
            for (int t = 0; t < counts[b]; ++t) {
                const auto previous =
                    std::uint64_t(t ? host[b * tokens + t - 1] : history[2 * b + 1]);
                const auto older = std::uint64_t(previous == 7 ? 7
                                                 : t >= 2      ? host[b * tokens + t - 2]
                                                               : history[2 * b + (t == 1)]);
                const auto pair =
                    std::uint64_t(host[b * tokens + t]) * weights.hash.multipliers[0] ^
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
        bytes += output.size() + options.batch * 4 + lookup_bytes;
        for (const auto& lut : used_lut) bytes += lut.count() * 2;
        ids.copy_from_host(host.data(), ids.size());
        tail.copy_from_host(history.data(), tail.size());
        valid.copy_from_host(counts.data(), valid.size());
        state_offsets.copy_from_host(select.data(), state_offsets.size());
    }

    void launch(CudaExecutionView execution) {
        ops::ngram_embedding_lookup(input, counts_view, before, table.hash, 7, table.shards,
                                    embedding, execution);
    }

    void restore(CudaExecutionView) {}

    std::uint64_t logical_bytes() const { return bytes; }
};
} // namespace ninfer::bench::ple
