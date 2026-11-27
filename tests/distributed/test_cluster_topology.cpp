#include "distributed/topology.h"
#if NINFER_ENABLE_DISTRIBUTED
#include "distributed/wire.h"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
namespace dist = ninfer::distributed;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class Fn>
void rejects(Fn&& fn, const char* message) {
    try {
        fn();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(message);
}

DeploymentOptions make_deployment(ParallelOptions parallel, std::uint32_t nodes_count = 1,
                                  bool shared_device = false) {
    DeploymentOptions d;
    d.coordinator_node    = 0;
    d.control_host        = "127.0.0.1";
    d.allow_shared_device = shared_device;
    d.parallel            = parallel;
    for (std::uint32_t n = 0; n < nodes_count; ++n) {
        DeploymentNode node;
        node.id       = n;
        node.ssh      = n == 0 ? "" : ("node" + std::to_string(n));
        node.worker   = "/usr/local/bin/ninfer-worker";
        node.artifact = "/models/model.ninfer";
        d.nodes.push_back(std::move(node));
    }
    const std::uint32_t world = parallel.dp * parallel.pp * parallel.tp;
    const std::uint32_t per_node = world / nodes_count;
    for (std::uint32_t r = 0; r < world; ++r) {
        const std::uint32_t node = r / per_node;
        const int dev            = shared_device ? 0 : static_cast<int>(r % per_node);
        d.ranks.push_back({r, node, dev});
    }
    return d;
}

void roundtrip_if_enabled(const DeploymentOptions& d, const dist::ExecutionTopology& t) {
#if NINFER_ENABLE_DISTRIBUTED
    dist::wire::Encoder out;
    dist::wire::encode(out, t);
    const auto bytes = std::move(out).take();
    dist::wire::Decoder in(bytes);
    const auto decoded = dist::wire::decode_topology(in);
    in.finish();
    decoded.validate(d);
    require(decoded.plan == t.plan && decoded.replicas.size() == t.replicas.size() &&
                decoded.ranks.size() == t.ranks.size(),
            "wire roundtrip lost topology fields");
    for (std::size_t i = 0; i < t.ranks.size(); ++i)
        require(decoded.ranks[i].shard_digest == t.ranks[i].shard_digest,
                "wire roundtrip lost rank shard_digest");
#else
    (void)d;
    (void)t;
#endif
}

void test_valid_1_to_16_card_topologies() {
    // 1-rank baseline
    {
        const auto d = make_deployment({.tp = 1, .ep = 1, .dp = 1, .pp = 1, .adp = 1});
        const auto t = dist::build_cluster_topology(d);
        t.validate(d);
        require(t.replicas.size() == 1 && t.replica(0).result_owner == 0 &&
                    t.replica(0).failure_domain_id == 0 &&
                    t.replica(0).local_index(0) == 0 && t.replica(0).global_rank(0) == 0,
                "1-rank baseline topology mismatch");
        roundtrip_if_enabled(d, t);
    }
    {
        const auto d = make_deployment({.tp = 2, .ep = 1, .dp = 1, .pp = 4, .adp = 1}, 2);
        const auto t = dist::build_cluster_topology(d);
        t.validate(d);
        for (const auto& rank : t.ranks) {
            const auto* input  = t.pipeline_input(rank.rank);
            const auto* output = t.pipeline_output(rank.rank);
            require(bool(input) == (rank.pp.rank != 0) && bool(output) == (rank.pp.rank != 3),
                    "pipeline endpoints do not match stage ownership");
            if (output) {
                const auto peer = t.replica(0).stage_ranks[rank.pp.rank + 1].at(rank.tp.rank);
                require(output == t.pipeline_input(peer) &&
                            output->members == std::vector<dist::RankId>{rank.rank, peer} &&
                            output->usage == dist::GroupUsage::PeerExchange,
                        "PP4/TP2 pipeline connected the wrong tensor lane");
            }
        }
        roundtrip_if_enabled(d, t);
    }

    {
        const auto d = make_deployment({.tp = 2, .dp = 2});
        auto t       = dist::build_cluster_topology(d);
        t.groups.push_back({1, t.replica(0).ranks});
        t.add_role("tensor", t.replica(0).ranks);
        t.add_role("tensor", t.replica(1).ranks);
        t.add_role("vocabulary", t.replica(1).ranks);
        t.add_role("context", t.replica(1).ranks, dist::GroupUsage::PeerExchange);
        t.validate(d);
        require(t.role_group("tensor", 0).id == 1 && t.role_group("tensor", 3).index(3) == 1 &&
                    t.role_group("tensor", 3).id == t.role_group("vocabulary", 3).id &&
                    t.role_group("tensor", 3).id != t.role_group("context", 3).id,
                "role lookup lost member order or communication usage");
        rejects([&] { (void)t.role_group("vocabulary", 1); },
                "rank borrowed another replica's role");
        roundtrip_if_enabled(d, t);
    }

    // 8-card flagship Decode: tp=2, dp=4, adp=2, ep=8, pp=1
    {
        const auto d = make_deployment({.tp = 2, .ep = 8, .dp = 4, .pp = 1, .adp = 2});
        const auto t = dist::build_cluster_topology(d);
        t.validate(d);
        require(t.replicas.size() == 4, "dp4 replica count");
        for (std::uint32_t r = 0; r < 4; ++r) {
            const auto& rep = t.replica(r);
            require(rep.replica_id == r && rep.failure_domain_id == 0 &&
                        rep.result_owner == 2 * r && rep.ranks.size() == 2 &&
                        rep.local_index(2 * r) == 0 && rep.local_index(2 * r + 1) == 1 &&
                        rep.global_rank(0) == 2 * r && rep.global_rank(1) == 2 * r + 1,
                    "flagship 8-card replica bijection mismatch");
        }
        require(t.rank(5).dp.rank == 2 && t.rank(5).tp.rank == 1 && t.rank(5).adp.rank == 1 &&
                    t.rank(5).ep.rank == 5 && t.rank(5).ep.size == 8,
                "flagship 8-card rank 5 coordinates mismatch");
        roundtrip_if_enabled(d, t);
    }

    // 8-card decoupled DP+TP+EP: tp=2, dp=4, ep=2, adp=1, pp=1
    {
        const auto d = make_deployment({.tp = 2, .ep = 2, .dp = 4, .pp = 1, .adp = 1});
        const auto t = dist::build_cluster_topology(d);
        t.validate(d);
        for (std::uint32_t r = 0; r < 4; ++r)
            require(t.replica(r).failure_domain_id == r, "decoupled DP must isolate failure domains");
        roundtrip_if_enabled(d, t);
    }

    // 8-card partial coupled EP: tp=2, dp=4, ep=4, adp=2, pp=1 -> 2 failure domains {0,0,1,1}
    {
        const auto d = make_deployment({.tp = 2, .ep = 4, .dp = 4, .pp = 1, .adp = 2});
        const auto t = dist::build_cluster_topology(d);
        t.validate(d);
        require(t.replica(0).failure_domain_id == 0 && t.replica(1).failure_domain_id == 0 &&
                    t.replica(2).failure_domain_id == 1 && t.replica(3).failure_domain_id == 1,
                "ep4 over tp2*dp4 must form 2 failure domains");
        require(t.rank(6).ep.rank == 2 && t.rank(7).ep.rank == 3,
                "domain-local EP coordinate mismatch");
    }

    // 8-card PP prefill: pp=2, tp=4, ep=2, dp=1, adp=2 (tests ep < tp intra-stage nesting)
    {
        const auto d = make_deployment({.tp = 4, .ep = 2, .dp = 1, .pp = 2, .adp = 2});
        const auto t = dist::build_cluster_topology(d);
        t.validate(d);
        const auto& rep = t.replica(0);
        require(rep.stage_ranks.size() == 2 && rep.stage_ranks[0].size() == 4 &&
                    rep.stage_ranks[1][0] == 4 && rep.result_owner == 4,
                "PP2*TP4 result_owner must be stage pp-1 tp 0 (global rank 4)");
        require(t.rank(5).pp.rank == 1 && t.rank(5).tp.rank == 1 && t.rank(5).ep.rank == 0 &&
                    t.rank(6).ep.rank == 1 && t.rank(7).adp.rank == 1,
                "PP2*TP4 intra-stage coordinates mismatch");
        roundtrip_if_enabled(d, t);
    }

    // 16-card 2x8 Decode & PP topologies
    {
        const auto decode16 =
            make_deployment({.tp = 2, .ep = 16, .dp = 8, .pp = 1, .adp = 2}, 2);
        const auto td = dist::build_cluster_topology(decode16);
        td.validate(decode16);
        require(td.rank(8).node == 1 && td.rank(8).local_rank == 0 && td.rank(15).ep.rank == 15 &&
                    td.replica(7).result_owner == 14 && td.replica(7).failure_domain_id == 0,
                "16-card 2x8 coupled EP topology mismatch");
        roundtrip_if_enabled(decode16, td);

        const auto pp16 = make_deployment({.tp = 8, .ep = 8, .dp = 1, .pp = 2, .adp = 1}, 2);
        const auto tp   = dist::build_cluster_topology(pp16);
        tp.validate(pp16);
        require(tp.replica(0).result_owner == 8 && tp.replica(0).local_index(15) == 15,
                "16-card pp2+tp8 topology mismatch");
        roundtrip_if_enabled(pp16, tp);
    }
}

void test_unordered_ranks_and_shared_device() {
    auto d       = make_deployment({.tp = 2, .ep = 2, .dp = 2, .pp = 1, .adp = 2}, 2);
    // Permute TOML rank order deliberately: {3, 1, 0, 2}
    d.ranks      = {{3, 1, 1}, {1, 0, 1}, {0, 0, 0}, {2, 1, 0}};
    const auto t = dist::build_cluster_topology(d);
    t.validate(d);

    const auto& rep1 = t.replica(1);
    require(rep1.result_owner == 2 && rep1.local_index(2) == 0 && rep1.local_index(3) == 1 &&
                rep1.global_rank(0) == 2 && rep1.global_rank(1) == 3 && !rep1.contains(0) &&
                rep1.contains(3),
            "unordered input broke replica 1 bijection");
    rejects([&] { (void)rep1.local_index(1); }, "foreign global rank accepted by local_index");
    rejects([&] { (void)rep1.global_rank(2); }, "out-of-range local_index accepted");

    // Shared GPU allowed vs rejected
    auto shared = make_deployment({.tp = 1, .ep = 1, .dp = 2, .pp = 1, .adp = 1}, 1, false);
    shared.ranks = {{0, 0, 0}, {1, 0, 0}};
    rejects([&] { validate_deployment(shared); },
            "duplicate (node, device) accepted when allow_shared_device=false");
    shared.allow_shared_device = true;
    validate_deployment(shared);
    shared.ranks[1].device = -1;
    rejects([&] { validate_deployment(shared); },
            "negative device accepted under allow_shared_device=true");
}

void test_invalid_combinations() {
    // dp > 1 && pp > 1
    rejects([&] { validate_deployment(make_deployment({2, 2, 2, 2, 1})); },
            "accepted dp > 1 && pp > 1");
    // pp > 1 && ep > tp
    rejects([&] { validate_deployment(make_deployment({2, 4, 1, 2, 1})); },
            "accepted pp > 1 && ep > tp");
    // tp % adp != 0
    rejects([&] { validate_deployment(make_deployment({4, 4, 1, 1, 3})); },
            "accepted tp % adp != 0");
    // non-nested ep / tp
    rejects([&] { validate_deployment(make_deployment({4, 3, 1, 1, 1})); },
            "accepted non-divisible ep/tp");
    // (dp * tp) % ep != 0 when ep > tp
    rejects([&] { validate_deployment(make_deployment({2, 8, 2, 1, 1})); },
            "accepted (dp * tp) % ep != 0");
    // world_size mismatch
    {
        auto d = make_deployment({2, 2, 2, 1, 1});
        d.ranks.pop_back();
        rejects([&] { validate_deployment(d); }, "accepted dp*pp*tp != ranks.size()");
    }
    // 64-bit multiplication overflow
    {
        DeploymentOptions d = make_deployment({1, 1, 1, 1, 1});
        d.parallel          = {.tp  = std::numeric_limits<std::uint32_t>::max(),
                               .ep  = 1,
                               .dp  = std::numeric_limits<std::uint32_t>::max(),
                               .pp  = std::numeric_limits<std::uint32_t>::max(),
                               .adp = 1};
        rejects([&] { validate_deployment(d); }, "accepted overflowing dp*pp*tp");
    }
}

void test_plan_digest_sensitivity() {
    const auto d = make_deployment({.tp = 2, .ep = 2, .dp = 2, .pp = 1, .adp = 2});
    auto base    = dist::build_cluster_topology(d);
    base.groups  = {{0, {0, 1}}, {1, {2, 3}}};
    base.roles   = {{"tensor", 0}, {"tensor", 1}};
    base.ranks[0].shard_digest = 101;
    base.ranks[2].shard_digest = 202;

    std::array<std::byte, 32> artifact_a{};
    artifact_a[0] = std::byte{0x42};
    std::array<std::byte, 32> artifact_b = artifact_a;
    artifact_b[31] = std::byte{0x99};

    const auto d0 = dist::compute_topology_plan_digest(artifact_a, base);
    base.plan = d0;
    roundtrip_if_enabled(d, base);
    require(d0 != 0, "plan digest must be nonzero");
    require(dist::compute_topology_plan_digest(artifact_b, base) != d0,
            "plan digest ignored artifact_id");

    auto changed_shard = base;
    changed_shard.ranks[2].shard_digest = 203;
    require(dist::compute_topology_plan_digest(artifact_a, changed_shard) != d0,
            "plan digest ignored rank shard_digest");

    auto changed_role = base;
    changed_role.roles[0].role = "vocabulary";
    require(dist::compute_topology_plan_digest(artifact_a, changed_role) != d0,
            "plan digest ignored communication roles");

    auto changed_owner = base;
    changed_owner.replicas[1].result_owner = 3;
    require(dist::compute_topology_plan_digest(artifact_a, changed_owner) != d0,
            "plan digest ignored replica result_owner");
}

} // namespace

int main() {
    try {
        test_valid_1_to_16_card_topologies();
        test_unordered_ranks_and_shared_device();
        test_invalid_combinations();
        test_plan_digest_sensitivity();
        std::cout << "PASS: cluster topology bijection, validation and plan digest\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
