#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/program/internal.h"

#include <cstdlib>
#include <iostream>

namespace {
using namespace ninfer;
namespace q4      = models::qwen4_exp;
namespace context = models::context;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void reclaim(q4::detail::ProgramImpl& p, DeviceContext& device) {
    const auto empty = p.usage().occupied;
    std::array<context::KVHistory, 2> owners;
    for (std::uint16_t pool = 0; pool < 2; ++pool) {
        auto& histories     = p.resources.pool(pool).histories;
        const auto frontier = pool == 0 ? 256U : 255U;
        auto binding        = histories.prepare_binding(nullptr, 0, frontier, 0);
        binding.enqueue_mapping(device.stream());
        device.synchronize();
        binding.publish_restores();
        owners[pool] = binding.commit();
        histories.prepare_write(owners[pool], 0, frontier, device.stream());
        histories.commit_frontier(owners[pool].handle(), frontier);
        const auto& physical = p.resources.pool(pool).cache.pages();
        for (std::size_t plane = 0; plane < physical.plane_count(); ++plane)
            CUDA_CHECK(cudaMemsetAsync(physical.plane(plane).data, 0, physical.plane(plane).bytes(),
                                       device.stream()));
    }
    const auto add = [&](std::uint32_t frontier) {
        auto& states     = *p.resources.state_store;
        const auto image = states.reserve_destination();
        require(image.has_value(), "state reservation failed");
        p.resources.states->storage().clear(states.physical_slot(*image), device.execution_view());
        states.complete_initialization(*image);
        states.freeze(*image);
        std::vector<context::KVPrefixRef> prefixes;
        for (std::uint16_t pool = 0; pool < 2; ++pool)
            prefixes.push_back(p.resources.pool(pool).histories.retain_prefix(owners[pool].handle(),
                                                                              frontier - pool));
        auto identity = std::make_shared<q4::detail::Identity>();
        models::qwen::PreparedPromptData prompt;
        prompt.token_ids.resize(frontier, 1);
        prompt.token_types.resize(frontier);
        prompt.positions.resize(frontier * 3);
        for (unsigned axis = 0; axis < 3; ++axis)
            for (unsigned i = 0; i < frontier; ++i) prompt.positions[axis * frontier + i] = i;
        identity->tokens = prompt.token_ids;
        identity->positions.assign(prompt);
        identity->digests.assign(prompt);
        const auto handle = *p.checkpoints.reserve();
        p.checkpoints.publish(
            handle,
            {q4::detail::Snapshot(
                 states, {*image, {.main_frontier = frontier, .mtp_frontier = frontier - 1}},
                 std::move(prefixes), std::move(identity)),
             runtime::CheckpointRole::SharedPrefix});
        return handle;
    };
    const auto shallow = add(128), deep = add(256);
    const auto alias = *p.checkpoints.reserve();
    p.checkpoints.publish(alias, p.checkpoints.at(shallow));
    const std::array all{shallow, deep, alias};
    device.synchronize();
    for (auto& owner : owners) owner.reset();
    const auto plan = p.plan_reclaim(all, {}, {.kv_page_groups = {4}});
    const auto tail =
        std::find_if(plan.demotions.begin(), plan.demotions.end(),
                     [&](const auto& quote) { return quote.sources == std::vector{deep}; });
    const auto family =
        std::find_if(plan.demotions.begin(), plan.demotions.end(),
                     [&](const auto& quote) { return quote.sources.size() == all.size(); });
    const auto tail_release =
        std::find_if(plan.releases.begin(), plan.releases.end(),
                     [&](const auto& quote) { return quote.sources == std::vector{deep}; });
    const auto family_release =
        std::find_if(plan.releases.begin(), plan.releases.end(),
                     [&](const auto& quote) { return quote.sources.size() == all.size(); });
    require(tail != plan.demotions.end() && family != plan.demotions.end() &&
                tail_release != plan.releases.end() && family_release != plan.releases.end(),
            "complete physical holder cohorts missing");
    auto partial = p.plan_reclaim(all, {}, {.kv_page_groups = {1}});
    const auto one_tail =
        std::find_if(partial.demotions.begin(), partial.demotions.end(),
                     [&](const auto& quote) { return quote.sources == std::vector{deep}; });
    require(one_tail != partial.demotions.end(), "suffix demotion missing");
    auto partial_batch = partial.begin_kv_batch({.kv_page_groups = {4}});
    require(partial_batch.append(*one_tail) && !partial_batch.covers(*tail_release),
            "partial physical move covered a two-page deletion by checkpoint name");
    auto batch = plan.begin_kv_batch({.kv_page_groups = {1}});
    require(batch.append(*tail) && batch.covers(*tail_release) && !batch.covers(*family_release) &&
                batch.append(*family) && batch.covers(*family_release),
            "batch coverage ignored actual page identities");
    auto wrong_pool = plan.begin_kv_batch({.kv_page_groups = {0, 1}});
    require(!wrong_pool.append(*tail), "Main quote entered an MTP batch");
    const auto clipped = batch.finish();
    require(clipped && clipped->released.kv_page_groups == std::array<std::uint32_t, 2>{1, 0} &&
                clipped->released.state_slots == 0,
            "batch did not clip to the requested typed shortage");

    const auto extra = *p.checkpoints.reserve();
    p.checkpoints.publish(extra, p.checkpoints.at(deep));
    require(!p.start_demote(*one_tail) && !batch.finish(),
            "stale quote ignored a newly published physical holder");
    require(p.checkpoints.erase(extra), "extra holder retirement failed");
    const auto pinned = one_tail->impl->pages[0].front();
    p.resources.pool(0).pages.pin_source(pinned);
    require(!p.start_demote(*one_tail), "demotion ignored a new KV reader");
    p.resources.pool(0).pages.unpin_source(pinned);
    auto altered = *one_tail;
    ++altered.host_bytes;
    require(!p.start_demote(altered), "demotion accepted a changed physical quote");

    const auto settle = [&] {
        for (unsigned attempt = 0; attempt < 8; ++attempt) {
            device.synchronize();
            CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream()));
            require(p.operation &&
                        std::holds_alternative<q4::detail::DemoteOperation>(p.operation->action),
                    "unexpected native transaction in demotion fixture");
            if (p.operation->migration.ready()) {
                p.operation->migration.publish();
                p.operation.reset();
                return;
            }
        }
        throw std::runtime_error("demotion did not complete");
    };
    auto current = p.plan_reclaim(all, {}, {.kv_page_groups = {1}});
    require(!current.demotions.empty(), "Main demotion quote missing");
    const auto before = p.usage().occupied;
    require(p.start_demote(current.demotions.front()), "Main demotion refused a valid quote");
    settle();
    auto after = p.usage().occupied;
    require(after.kv_page_groups[0] + 1 == before.kv_page_groups[0] &&
                after.kv_page_groups[1] == before.kv_page_groups[1] &&
                after.state_slots == before.state_slots,
            "Main demotion changed another physical pool");
    require(!p.start_demote(current.demotions.front()), "stale residency quote was reused");
    auto mtp = p.plan_reclaim(all, {}, {.kv_page_groups = {0, 1}});
    require(!mtp.demotions.empty() && p.start_demote(mtp.demotions.front()),
            "MTP demotion missing");
    settle();
    const auto after_mtp = p.usage().occupied;
    require(after_mtp.kv_page_groups[1] + 1 == after.kv_page_groups[1] &&
                after_mtp.kv_page_groups[0] == after.kv_page_groups[0],
            "MTP demotion used Main resources");
    auto state = p.plan_reclaim(all, {}, {.state_slots = 1});
    require(!state.demotions.empty() && p.start_demote(state.demotions.front()),
            "State demotion missing");
    settle();
    const auto after_state = p.usage().occupied;
    require(after_state.state_slots + 1 == after_mtp.state_slots &&
                after_state.kv_page_groups == after_mtp.kv_page_groups,
            "State demotion changed KV residency");
    auto host              = p.plan_releases(all, {}, {.host_bytes = 1});
    auto selected          = host.inventory->select();
    auto probe             = selected.probe(all);
    const auto quoted_host = selected.released().host_bytes;
    p.checkpoints.clear();
    const auto released = p.usage().occupied;
    require(after_state.host_bytes - released.host_bytes == quoted_host && released == empty,
            "fixed physical release differed from actual Native occupancy");
}
} // namespace

int main() {
    const char* path = std::getenv("NINFER_QWEN4_EXP_TEST_ARTIFACT");
    if (!path || !*path) return 77;
    try {
        int devices      = 0;
        const auto error = cudaGetDeviceCount(&devices);
        if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || !devices)
            return 77;
        CUDA_CHECK(error);
        DeviceContext device(0);
        EngineOptions options;
        options.use_cuda_graph                    = false;
        options.enable_vision                     = false;
        options.max_context                       = 512;
        options.prefill_chunk                     = 256;
        options.max_concurrency                   = 1;
        options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(512);
        options.speculative.backend               = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens          = 1;
        options.context_cache.device_state_slots  = 4;
        options.context_cache.host_capacity_bytes = 256ULL * 1024 * 1024;
        artifact::Reader reader(path);
        auto load      = q4::plan_load(reader, models::load_options(options));
        auto profile   = q4::execution::describe_model(load);
        auto execution = q4::plan_execution(profile, load.ple_table().geometry(), options, device,
                                            8ULL * 1024 * 1024 * 1024,
                                            q4::describe_moe_plan(*profile, options.moe));
        require(execution.has_value(), "reclaim test execution plan does not fit");
        auto model = q4::materialize_model(
            q4::resolve_load(std::move(load), device.device(), Device::cpu()), device);
        q4::execution::Parameters parameters(*model, profile);
        q4::detail::ProgramImpl program(parameters, std::move(*execution), device);
        reclaim(program, device);
        std::cout << "OK Qwen4Exp recovery quotes, physical coverage and migrations\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
