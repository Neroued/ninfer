#include "ninfer/engine.h"
#include "product/config/options.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {
using namespace ninfer;

void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}

void run(const char* artifact, std::string_view topology, std::string_view mode,
         std::string_view expected) {
    EngineOptions options;
    options.artifact_path = artifact;
    if (topology != "local")
        options.deployment = product::read_deployment_file(std::string(topology));
    options.purpose                = EnginePurpose::CausalScoring;
    options.max_context            = 512;
    options.prefill_chunk          = 128;
    constexpr std::size_t capacity = 2ULL << 30;
    options.context_capacity       = ContextCapacityPolicy::explicit_capacity(capacity);
    options.use_cuda_graph         = false;
    const auto consumers           = options.deployment ? options.deployment->parallel.tp : 1U;
    if (mode == "disk")
        options.ple.storage = PleStorage::Disk;
    else if (mode == "replicated")
        options.ple.device_layout = PleDeviceLayout::Replicated;
    else if (mode == "sharded")
        options.ple.device_layout = PleDeviceLayout::Sharded;
    else if (mode == "latency")
        options.ple.preference = PlePreference::Latency;
    else
        require(mode == "capacity", "unknown PLE placement");
    Engine engine(options);
    std::vector<TokenId> input(261);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = 151 + (i * 313) % 19003;
    const auto scores = engine.score_tokens(input, 126);
    require(scores.size() == 135 &&
                std::ranges::all_of(scores,
                                    [](float value) { return std::isfinite(value) && value <= 0; }),
            "causal scoring failed across three PLE windows");
    engine.reset_measurement_peaks();
    const auto snap = engine.snapshot();
    require(snap.ranks.size() == consumers, "fixture requires one TP replica");
    std::uint32_t next_shard = 0;
    for (const auto& rank : snap.ranks) {
        require(rank.memory.ple_table.has_value(), "rank omitted PLE placement");
        const auto& table = *rank.memory.ple_table;
        require(table.total_shards && table.local_shards <= table.total_shards,
                "invalid PLE range");
        if (!expected.empty())
            require(table.placement == expected,
                    "PLE selection differs from the expected feasible layout");
        require(table.merged_windows == (table.placement == "device/sharded" ? 3U : 0U),
                "actual PLE merge count differs from the three scoring windows");
        const auto& allocations = rank.memory.allocations;
        require(allocations.context_device_bytes == capacity,
                "placement changed explicit Context capacity");
        if (mode == "replicated" || (mode == "sharded" && consumers == 1))
            require(table.placement == "device/replicated", "explicit replicated PLE was lost");
        if (mode == "disk") require(table.placement == "disk", "explicit Disk PLE was lost");
        if (mode == "sharded" && consumers > 1)
            require(table.placement == "device/sharded", "explicit sharded PLE was lost");
        if (table.placement == "disk") {
            require(!table.local_shards && !table.device_bytes && table.host_bytes &&
                        allocations.ple_host_bytes >= table.host_bytes &&
                        allocations.ple_pinned_bytes && rank.memory.ple &&
                        rank.memory.ple->read_bytes && rank.memory.ple->uploaded_bytes,
                    "Disk placement did not own and use its Host reader and upload slots");
        } else {
            require(!rank.memory.ple && !allocations.ple_pinned_bytes &&
                        allocations.ple_device_bytes == table.device_bytes,
                    "incorrect device PLE ledger");
            if (table.placement == "device/replicated")
                require(!table.first_shard && table.local_shards == table.total_shards,
                        "incomplete PLE replica");
            else {
                require(table.placement == "device/sharded" && table.first_shard == next_shard,
                        "sharded PLE coverage failed");
                next_shard += table.local_shards;
            }
        }
        std::cout << "rank=" << rank.rank << " placement=" << table.placement
                  << " shard=" << table.first_shard << "+" << table.local_shards << "/"
                  << table.total_shards << " table_device=" << table.device_bytes
                  << " merged_windows=" << table.merged_windows
                  << " base_weights=" << allocations.device_weights_bytes
                  << " table_host=" << table.host_bytes
                  << " ple_device=" << allocations.ple_device_bytes
                  << " ple_pinned=" << allocations.ple_pinned_bytes
                  << " ple_host=" << allocations.ple_host_bytes
                  << " context=" << allocations.context_device_bytes << '\n';
    }
    if (snap.ranks.front().memory.ple_table->placement == "device/sharded")
        require(next_shard == snap.ranks.front().memory.ple_table->total_shards,
                "missing final PLE shard");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 5)
            throw std::invalid_argument(
                "expected MODEL local|DEPLOYMENT disk|replicated|sharded|capacity|latency");
        run(argv[1], argv[2], argv[3], argc == 5 ? argv[4] : "");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
