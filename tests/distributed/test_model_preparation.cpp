#include "models/qwen4_exp/parallel/preparation.h"
#include "models/qwen4_exp/startup.h"
#include "models/qwen4_exp/parallel/wire.h"
#include "product/config/options.h"
#include "runtime/executor/preparation.h"

#include <iostream>

int main(int argc, char** argv) {
    using namespace ninfer;
    namespace q4   = models::qwen4_exp;
    namespace wire = distributed::wire;
    if (argc != 3) return 77;
    try {
        auto deployment = product::read_deployment_file(argv[1]);
        artifact::Reader reader(argv[2]);
        EngineOptions options;
        options.use_cuda_graph                    = false;
        options.enable_vision                     = true;
        options.max_context                       = 4096;
        options.max_concurrency                   = 2;
        options.prefill_chunk                     = 4096;
        options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(8192);
        options.context_cache.host_capacity_bytes = 0;
        options.speculative = {.backend = SpeculativeBackend::Mtp, .draft_tokens = 3};
        auto definition     = q4::read_definition(reader, models::load_options(options));
        auto frontend       = q4::create_frontend(*definition, options);
        const auto plan     = q4::plan_parallel(*definition, deployment);
        std::vector<wire::Bytes> payloads;
        for (const auto& rank : plan.ranks) {
            wire::Encoder out;
            const auto node = plan.topology.rank(rank.rank).node;
            q4::encode_rank_load(
                out, {definition->artifact_id(), rank, options,
                      node == deployment.coordinator_node ? options.media_live_bytes : 0});
            payloads.push_back(std::move(out).take());
            wire::Decoder in(payloads.back());
            const auto decoded = q4::decode_rank_load(in);
            in.finish();
            if (decoded.rank.experts != rank.experts ||
                decoded.rank.assemblies.size() != rank.assemblies.size())
                throw std::runtime_error("native rank plan did not round-trip");
        }
        distributed::ExecutionGroup group(deployment, plan.topology,
                                          [](auto rank, std::string_view text) {
                                              std::cerr << "[rank " << rank << "] " << text << '\n';
                                          });
        auto policy                     = options.kv_capacity;
        policy.automatic_headroom_bytes = 0;
        const auto result =
            runtime::prepare_execution_group(group, payloads, policy, std::chrono::minutes(15));
        for (std::size_t i = 0; i < result.reports.size(); ++i) {
            const auto& r = result.reports[i];
            wire::Decoder in(result.layouts[i]);
            const auto layout = q4::decode_rank_layout(in);
            in.finish();
            if (layout.main_pages != result.pages || layout.gdn_key_heads != 8 ||
                layout.gdn_value_heads != 24 || layout.qsa_query_heads != 12 ||
                layout.qsa_kv_heads != 1 || layout.ple_shards != 64 ||
                layout.vocabulary != 124160 || layout.vocabulary_begin != i * 124160 ||
                r.host_weight_bytes)
                throw std::runtime_error("prepared local model layout differs from TP2/EP2 plan");
            std::cout << "rank=" << i << " weights=" << r.weight_bytes
                      << " runtime=" << layout.physical_bytes << " state=" << layout.state_bytes
                      << " slots=" << layout.state_slots << " main_pages=" << layout.main_pages
                      << " mtp_pages=" << layout.mtp_pages
                      << " workspace=" << layout.workspace_bytes
                      << " available=" << r.capacity.available_bytes
                      << " staging=" << r.staging_bytes << " read=" << r.read_bytes
                      << " h2d=" << r.upload_bytes << " load_seconds=" << r.load_seconds
                      << std::endl;
        }
        group.shutdown();
        std::cout << "PASS: real distributed preparation and ordered shutdown\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
