#pragma once

#include "core/buffer.h"
#include "ninfer/ops/candidate_selector.h"

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <set>
#include <vector>

namespace ninfer::tool {

// Represented inputs only. Mathematical selector references live in the tests.
struct GatheredSelectorFixture {
    static constexpr int Rank = 256, Candidates = 16, Rows = 154880, Domain = 154856;
    enum class Pattern { Regular, Ties, Precision, EmptyRank, Missing };

    struct Candidate {
        int id;
        float score;
    };

    Device device;
    int steps, batch, generation = 0;
    Pattern pattern;
    std::vector<unsigned long long> key_words;
    std::vector<int> ids, anchors, positions, touched;
    std::vector<float> unary;
    std::vector<std::uint16_t> hidden_bits;
    std::vector<ops::SamplingConfig> sampling;
    DeviceBuffer keys, hidden, anchor, position, configs, predecessor, successor;

    static std::uint32_t mix(std::uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        return x ^ (x >> 16);
    }

    static std::uint16_t bf16(float x) {
        auto bits = std::bit_cast<std::uint32_t>(x);
        return std::uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
    }

    static float value(std::uint16_t bits) {
        return std::bit_cast<float>(std::uint32_t(bits) << 16);
    }

    static std::uint16_t weight(int token, int feature, bool pred) {
        const int x = int(mix(std::uint32_t(token) * 7919U ^ std::uint32_t(feature) * 104729U ^
                              (pred ? 101U : 211U)) %
                          257) -
                      128;
        return bf16(x / 256.F);
    }

    static unsigned long long pack(float score, int token) {
        if (score == 0) score = 0;
        auto bits = std::bit_cast<std::uint32_t>(score);
        bits      = bits >> 31 ? ~bits : bits ^ 0x80000000U;
        return (static_cast<unsigned long long>(bits) << 32) | (0xffffffffU - unsigned(token));
    }

    GatheredSelectorFixture(Device d, int k, int b, Pattern p = Pattern::Regular)
        : device(d), steps(k), batch(b), pattern(p), keys(d, 2ULL * 16 * k * b * 8),
          hidden(d, 256ULL * k * b * 2), anchor(d, b * 4), position(d, b * 4),
          configs(d, b * sizeof(ops::SamplingConfig)), predecessor(d, 256ULL * Rows * 2),
          successor(d, 256ULL * Rows * 2) {
        predecessor.fill(0xff);
        successor.fill(0xff);
        refresh(0);
    }

    void refresh(int salt) {
        generation        = salt;
        const int columns = steps * batch;
        key_words.assign(2 * 16 * columns, 0);
        ids.assign(16 * columns, INT_MAX);
        unary.assign(16 * columns, -INFINITY);
        anchors.resize(batch);
        positions.resize(batch);
        hidden_bits.resize(256 * columns);
        std::set<int> rows;
        for (int b = 0; b < batch; ++b) {
            anchors[b]   = b % 2 ? Domain - 1 - b * 101 : 17 + b * 97;
            positions[b] = 1'048'560 + b * 97 + salt * 137;
            rows.insert(anchors[b]);
        }
        for (int column = 0; column < columns; ++column) {
            std::vector<Candidate> all;
            for (int rank = 0; rank < 2; ++rank) {
                const int begin = rank * 77440, count = rank ? Domain - begin : 77440;
                int live = pattern == Pattern::EmptyRank && rank == 0 ? 0
                           : pattern == Pattern::Missing              ? (rank ? 5 : 3)
                                                                      : 16;
                std::vector<Candidate> local;
                std::set<int> unique;
                for (int c = 0; c < live; ++c) {
                    int token =
                        begin +
                        int((std::uint64_t(column + salt * 31) * 7919 + c * 4099 + 17) % count);
                    if (column == 0 && c == 0) token = rank ? 77440 : 77439;
                    if (column == 0 && c == 1) token = rank ? Domain - 1 : 0;
                    while (unique.contains(token)) token = begin + (token - begin + 1) % count;
                    unique.insert(token);
                    rows.insert(token);
                    float score =
                        (int(mix(unsigned(column * 37 + c * 101 + rank * 997 + salt * 19)) % 513) -
                         256) /
                        512.F;
                    if (pattern == Pattern::Ties) score = 0;
                    if (pattern == Pattern::Precision)
                        score = 1.F + float(32 - c * 2 - rank) * 0x1p-23F;
                    local.push_back({token, score});
                }
                const auto greater = [](Candidate a, Candidate b) {
                    return a.score > b.score || (a.score == b.score && a.id < b.id);
                };
                std::sort(local.begin(), local.end(), greater);
                for (int c = 0; c < live; ++c)
                    key_words[(rank * columns + column) * 16 + c] =
                        pack(local[c].score, local[c].id);
                all.insert(all.end(), local.begin(), local.end());
            }
            std::sort(all.begin(), all.end(), [](Candidate a, Candidate b) {
                return a.score > b.score || (a.score == b.score && a.id < b.id);
            });
            for (int c = 0; c < std::min<int>(16, all.size()); ++c) {
                ids[column * 16 + c]   = all[c].id;
                unary[column * 16 + c] = all[c].score;
            }
            for (int r = 0; r < 256; ++r)
                hidden_bits[column * 256 + r] =
                    pattern == Pattern::Ties || pattern == Pattern::Precision
                        ? 0
                        : bf16((int(mix(unsigned(column * 257 + r + salt * 999)) % 257) - 128) /
                               512.F);
        }
        touched.assign(rows.begin(), rows.end());
        std::vector<std::uint16_t> row(256);
        for (int token : touched)
            for (bool pred : {true, false}) {
                for (int r = 0; r < 256; ++r) row[r] = weight(token, r, pred);
                (pred ? predecessor : successor)
                    .copy_from_host(row.data(), 512, std::size_t(token) * 512);
            }
        keys.copy_from_host(key_words.data(), key_words.size() * 8);
        hidden.copy_from_host(hidden_bits.data(), hidden_bits.size() * 2);
        anchor.copy_from_host(anchors.data(), batch * 4);
        position.copy_from_host(positions.data(), batch * 4);
        set_mode(2);
    }

    void set_mode(int mode) {
        sampling.assign(batch, {});
        for (int b = 0; b < batch; ++b) {
            auto& c             = sampling[b];
            c.temperature       = mode == 0 || (mode == 2 && b % 2) ? 0.F
                                  : pattern == Pattern::Precision   ? 1.e-6F
                                                                    : .8F;
            c.seed              = 0xa55a123400ULL + b * 7919 + generation * 31;
            c.top_k             = 1;
            c.top_p             = .001F;
            c.min_p             = .99F;
            c.presence_penalty  = 100;
            c.frequency_penalty = 200;
            c.token_counts      = reinterpret_cast<const int*>(std::uintptr_t(0x1234));
            c.mask = {reinterpret_cast<const std::uint32_t*>(std::uintptr_t(0x4320)), 0};
        }
        configs.copy_from_host(sampling.data(), sampling.size() * sizeof(ops::SamplingConfig));
    }

    Tensor key_tensor() { return {keys.span(), DType::I64, {16, steps, batch, 2}}; }

    Tensor hidden_tensor() { return {hidden.span(), DType::BF16, {256, steps, batch}}; }

    Tensor anchor_tensor() { return {anchor.span(), DType::I32, {batch}}; }

    Tensor position_tensor() { return {position.span(), DType::I32, {batch}}; }

    Tensor predecessor_tensor() { return {predecessor.span(), DType::BF16, {256, Rows}}; }

    Tensor successor_tensor() { return {successor.span(), DType::BF16, {256, Rows}}; }
};
} // namespace ninfer::tool
