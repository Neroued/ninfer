#pragma once
#include "core/buffer.h"
#include "ninfer/ops/vocabulary.h"
#include <algorithm>
#include <bit>
#include <climits>
#include <vector>

namespace ninfer::tool {
struct SparseVocabularyFixture {
    struct Candidate {
        int id, rank;
        float score;
    };
    enum class Pattern { Regular, Ratio, EmptyRank, DenseQ };
    int steps, batch, ranks, mode, accepted;
    Pattern pattern;
    std::vector<std::uint64_t> words;
    std::vector<std::vector<Candidate>> targets;
    std::vector<int> drafts, ids, extents, positions;
    std::vector<float> q;
    std::vector<ops::SamplingConfig> configs;
    DeviceBuffer key_buffer, draft_buffer, id_buffer, q_buffer, extent_buffer, position_buffer,
        config_buffer;

    static std::uint64_t pack(float score, int token) {
        if (score == 0) score = 0;
        auto b = std::bit_cast<std::uint32_t>(score);
        b      = b >> 31 ? ~b : b ^ 0x80000000U;
        return (std::uint64_t(b) << 32) | (0xffffffffU - unsigned(token));
    }

    static bool greater(Candidate a, Candidate b) {
        return a.score > b.score || (a.score == b.score && a.id < b.id);
    }

    SparseVocabularyFixture(Device device, int d, int b, int r = 2, int m = 1, int a = -1,
                            Pattern p = Pattern::Regular)
        : steps(d), batch(b), ranks(r), mode(m), accepted(a < 0 ? d : a), pattern(p),
          key_buffer(device, 20ULL * (d + 1) * b * r * 8), draft_buffer(device, d * b * 4),
          id_buffer(device, 16ULL * d * b * 4), q_buffer(device, 16ULL * d * b * 4),
          extent_buffer(device, b * 4), position_buffer(device, b * 4),
          config_buffer(device, b * sizeof(ops::SamplingConfig)) {
        refresh(0, false);
    }

    void refresh(int phase, bool varied_extents) {
        words.assign(20 * (steps + 1) * batch * ranks, ~0ULL);
        drafts.assign(steps * batch, INT_MIN);
        ids.assign(16 * steps * batch, INT_MIN);
        q.assign(ids.size(), std::bit_cast<float>(0x7fc00001U));
        targets.assign((steps + 1) * batch, {});
        extents.resize(batch);
        positions.resize(batch);
        configs.assign(batch, {});
        for (int b = 0; b < batch; ++b) {
            extents[b]      = varied_extents && b + 1 < batch ? (b + phase) % (steps + 1) : steps;
            positions[b]    = 1'048'570 + b * 97 + phase * 131;
            auto& cfg       = configs[b];
            cfg.temperature = mode == 0 || (mode == 2 && b % 2) ? 0 : .75F;
            cfg.seed        = 0x541208aULL + b * 7919 + phase * 37;
            cfg.top_k       = pattern == Pattern::Ratio ? 20
                                                        : (varied_extents && b % 3 == 0 ? 1
                                                           : b % 3 == 1                 ? 7
                                                                                        : 20);
            cfg.top_p = pattern == Pattern::Ratio ? 1 : (varied_extents && b % 3 == 1 ? .63F : 1);
            cfg.min_p = pattern == Pattern::Ratio ? 0 : (varied_extents && b % 3 == 2 ? .4F : 0);
            cfg.presence_penalty  = 100;
            cfg.frequency_penalty = 200;
            cfg.token_counts      = reinterpret_cast<const int*>(std::uintptr_t(0x1234));
            cfg.mask = {reinterpret_cast<const std::uint32_t*>(std::uintptr_t(0x4320)), 0};
            for (int col = 0; col <= extents[b]; ++col) {
                auto& all = targets[b * (steps + 1) + col];
                for (int r = 0; r < ranks; ++r) {
                    const int count = pattern == Pattern::EmptyRank && r == 0 ? 0
                                      : pattern == Pattern::Ratio             ? 2
                                                                              : 20;
                    std::vector<Candidate> local;
                    for (int c = 0; c < count; ++c) {
                        int token = r * 9000 + 100 + c;
                        if (ranks == 2)
                            token = r ? (c == 0   ? 77440
                                         : c == 1 ? 154855
                                                  : 80000 + c)
                                      : (c == 0   ? 77439
                                         : c == 1 ? 0
                                                  : 100 + c);
                        float score =
                            pattern == Pattern::Ratio
                                ? 0
                                : (20 - c) * .125F + float((r + col + phase) % 3) * .03125F;
                        local.push_back({token, r, score});
                    }
                    std::sort(local.begin(), local.end(), greater);
                    auto at = ((r * batch + b) * (steps + 1) + col) * 20;
                    std::fill_n(words.begin() + at, 20, 0);
                    for (int c = 0; c < count; ++c)
                        words[at + c] = pack(local[c].score, local[c].id);
                    all.insert(all.end(), local.begin(), local.end());
                }
                std::sort(all.begin(), all.end(), greater);
                if (col == extents[b]) continue;
                const int outside       = 150000 + b * 16 + col;
                const bool reject       = col == accepted;
                const int chosen        = pattern == Pattern::Ratio ? all[0].id
                                          : reject                  ? outside
                                                                    : all[0].id;
                drafts[b * steps + col] = chosen;
                const int at            = (b * steps + col) * 16;
                std::fill_n(ids.begin() + at, 16, INT_MAX);
                std::fill_n(q.begin() + at, 16, 0.F);
                const auto set = [&](int c, int id, float probability) {
                    const int slot = (c * 5 + 7) % 16;
                    ids[at + slot] = id;
                    q[at + slot]   = probability;
                };
                if (pattern == Pattern::DenseQ) {
                    drafts[b * steps + col] = all[0].id;
                    for (int c = 0; c < 16; ++c) set(c, all[c].id, 1.F / 16);
                } else if (pattern == Pattern::Ratio) {
                    set(0, chosen, .5F);
                    set(1, outside, .5F);
                } else if (reject) {
                    set(0, chosen, .375F);
                    set(1, all[0].id, .25F);
                    set(2, all[1].id, .375F);
                } else {
                    set(0, chosen, .0001F);
                    set(1, all[1].id, .25F);
                    set(2, outside, .7499F);
                }
            }
        }
        key_buffer.copy_from_host(words.data(), words.size() * 8);
        draft_buffer.copy_from_host(drafts.data(), drafts.size() * 4);
        id_buffer.copy_from_host(ids.data(), ids.size() * 4);
        q_buffer.copy_from_host(q.data(), q.size() * 4);
        extent_buffer.copy_from_host(extents.data(), batch * 4);
        position_buffer.copy_from_host(positions.data(), batch * 4);
        config_buffer.copy_from_host(configs.data(), configs.size() * sizeof(ops::SamplingConfig));
    }

    Tensor keys() { return {key_buffer.span(), DType::I64, {20, steps + 1, batch, ranks}}; }

    Tensor draft_view() { return {draft_buffer.span(), DType::I32, {steps, batch}}; }

    Tensor id_view() { return {id_buffer.span(), DType::I32, {16, steps, batch}}; }

    Tensor q_view() { return {q_buffer.span(), DType::FP32, {16, steps, batch}}; }

    Tensor extent_view() { return {extent_buffer.span(), DType::I32, {batch}}; }

    Tensor position_view() { return {position_buffer.span(), DType::I32, {batch}}; }
};
} // namespace ninfer::tool
