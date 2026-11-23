#pragma once

#include "ninfer/observability/snapshot.h"

#include <map>
#include <span>
#include <stdexcept>
#include <tuple>

namespace ninfer::test {

// For one positive-width bucket the Target, Decision and Commit payloads are fixed. Subtract
// the mathematical proposal collectives before comparing rounds at different depths.
class MtpGraphTraffic {
public:
    void begin(std::span<const RankSnapshot> ranks) {
        before_.clear();
        for (const auto& rank : ranks) before_.push_back(rank.communication_graph);
    }

    void finish(std::span<const RankSnapshot> ranks, const SpeculativeExecutionGeometry& g,
                unsigned hidden, bool attention_dp) {
        if (ranks.size() != 2 || g.path != ExecutionPath::Graph || !g.mtp_steps) return;
        const std::uint64_t d = g.mtp_steps, p = g.physical_rows, l = g.owner_rows;
        for (std::size_t r = 0; r < ranks.size(); ++r) {
            auto fixed = ranks[r].communication_graph;
            for (std::size_t op = 0; op < fixed.calls.size(); ++op) {
                fixed.calls[op] -= before_[r].calls[op];
                fixed.payload_bytes[op] -= before_[r].payload_bytes[op];
            }
            const auto subtract = [&](unsigned op, std::uint64_t calls, std::uint64_t bytes) {
                if (fixed.calls[op] < d * calls || fixed.payload_bytes[op] < d * bytes)
                    throw std::runtime_error("MTP graph omitted proposal communication");
                fixed.calls[op] -= d * calls;
                fixed.payload_bytes[op] -= d * bytes;
            };
            subtract(0, attention_dp ? 1 : 2, (attention_dp ? 2ULL : 4ULL) * hidden * p);
            subtract(2, 1, 16 * p);
            if (attention_dp) subtract(2, 1, 2ULL * hidden * l);
            const auto [it, inserted] = fixed_.try_emplace({r, p, l, g.verify_width}, fixed);
            if (!inserted && (it->second.calls != fixed.calls ||
                              it->second.payload_bytes != fixed.payload_bytes))
                throw std::runtime_error(
                    "MTP graph communication does not match the executed depth");
        }
    }
private:
    std::vector<CommunicationWork> before_;
    std::map<std::tuple<std::size_t, unsigned, unsigned, unsigned>, CommunicationWork> fixed_;
};

} // namespace ninfer::test
