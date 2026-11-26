#pragma once

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ninfer::test::softcap {

inline constexpr std::array distributions{"uniform", "reuse", "pool", "tail", "spread"};

inline std::vector<int> make_routes(std::string_view distribution, int tokens, int experts,
                                    int top_k) {
    std::mt19937 random(236817);
    std::vector<int> ids(std::size_t(tokens) * top_k);
    std::vector<int> pool(experts);
    std::iota(pool.begin(), pool.end(), 0);
    // Reserve the upper half of the experts for one assignment each. The lower
    // half contains the permanently hot experts and a Zipf-distributed middle.
    const int hot = top_k / 2, cutoff = experts / 2;
    std::vector<double> weights(cutoff - hot);
    for (std::size_t j = 0; j < weights.size(); ++j) weights[j] = 1.0 / (j + 1);
    std::discrete_distribution<int> zipf(weights.begin(), weights.end());
    for (int t = 0; t < tokens; ++t) {
        const auto begin = ids.begin() + std::size_t(t) * top_k;
        if (distribution == "spread") {
            for (int j = 0; j < top_k; ++j) begin[j] = (t * top_k + j) % experts;
        } else if (distribution == "reuse") {
            std::iota(begin, begin + top_k, 0);
        } else if (distribution == "uniform" || distribution == "pool") {
            const int domain = distribution == "pool" ? 2 * top_k : experts;
            for (int j = 0; j < top_k; ++j) {
                const int slot = std::uniform_int_distribution<int>(j, domain - 1)(random);
                std::swap(pool[j], pool[slot]);
                begin[j] = pool[j];
            }
        } else if (distribution == "tail") {
            std::iota(begin, begin + hot, 0);
            int j          = hot;
            const int cold = cutoff + t * (top_k - hot);
            for (; j < top_k && cold + j - hot < experts; ++j) begin[j] = cold + j - hot;
            for (; j < top_k; ++j) {
                int expert;
                do {
                    expert = hot + zipf(random);
                } while (std::find(begin, begin + j, expert) != begin + j);
                begin[j] = expert;
            }
        } else {
            throw std::invalid_argument("unknown route distribution");
        }
    }
    return ids;
}

} // namespace ninfer::test::softcap
