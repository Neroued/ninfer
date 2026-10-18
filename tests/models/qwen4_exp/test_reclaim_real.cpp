#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/runtime_types.h"
#include "models/qwen4_exp/session/internal.h"
#include "runtime/executor/local_executor.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
namespace q4      = models::qwen4_exp;
namespace context = models::context;
using Executor    = runtime::LocalExecutor<q4::RuntimeTypes>;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

q4::ContextResult retire_context(Executor& executor, runtime::ExecutionTicket ticket) {
    std::optional<q4::ContextResult> result;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
        auto event = executor.poll();
        if (!event) {
            std::this_thread::yield();
            continue;
        }
        if (const auto* failure = std::get_if<runtime::ExecutionFailed>(&*event))
            throw std::runtime_error(failure->message);
        require(std::visit([&](const auto& value) { return value.ticket == ticket; }, *event),
                "context fixture received another execution's event");
        if (auto* ready = std::get_if<runtime::ResultReady<q4::ExecutionTypes>>(&*event)) {
            require(!result, "context fixture received a repeated result");
            result.emplace(std::get<q4::ContextResult>(std::move(ready->result)));
        } else if (std::holds_alternative<runtime::DeviceRetired>(*event)) {
            require(result.has_value(), "context fixture retired before its result");
            return std::move(*result);
        } else {
            throw std::runtime_error("context fixture requested a sampling decision");
        }
    }
    throw std::runtime_error("context fixture did not retire before its deadline");
}

void execute(Executor& executor, context::ContextTransaction& transaction) {
    transaction.prepare_actions();
    auto reservation = executor.try_reserve(runtime::FrameKind::Context);
    require(reservation.has_value(), "context fixture execution slot is occupied");
    const auto ticket =
        executor.submit(std::move(*reservation), {transaction.actions(), q4::ContextWork{}});
    transaction.mark_submitted();
    try {
        (void)retire_context(executor, ticket);
        transaction.mark_completed();
        transaction.publish();
    } catch (...) {
        try {
            executor.drain();
        } catch (...) {}
        if (transaction.in_flight()) transaction.mark_completed();
        transaction.discard();
        throw;
    }
}

void reclaim(q4::detail::SessionImpl& p, Executor& executor) {
    const auto empty = p.usage().occupied;
    std::array<context::HistoryOwner, 2> owners;
    {
        auto bindings = p.directory.begin_transaction();
        for (std::uint16_t pool = 0; pool < 2; ++pool)
            bindings.add_binding(pool, p.directory.pool(pool).histories().prepare_binding(
                                           nullptr, 0, 256U - pool, 0));
        execute(executor, bindings);
        for (std::uint16_t pool = 0; pool < 2; ++pool)
            owners[pool] = bindings.binding(pool)->commit();
    }
    {
        auto writes = p.directory.begin_transaction();
        for (std::uint16_t pool = 0; pool < 2; ++pool)
            p.directory.pool(pool).histories().prepare_write(owners[pool], 0, 256U - pool, writes);
        execute(executor, writes);
        for (std::uint16_t pool = 0; pool < 2; ++pool)
            p.directory.pool(pool).histories().commit_frontier(owners[pool].handle(), 256U - pool);
    }
    const auto add = [&](std::uint32_t frontier) {
        auto& states     = p.directory.states();
        const auto image = states.reserve_destination();
        require(image.has_value(), "state reservation failed");
        auto initialization = p.directory.begin_transaction();
        initialization.own_state(*image);
        initialization.clear_state();
        execute(executor, initialization);
        states.complete_initialization(*image);
        (void)initialization.state_destination().release();
        states.freeze(*image);
        std::vector<context::HistoryPrefix> prefixes;
        for (std::uint16_t pool = 0; pool < 2; ++pool)
            prefixes.push_back(p.directory.pool(pool).histories().retain_prefix(
                owners[pool].handle(), frontier - pool));
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
    p.directory.pool(0).pages().pin_source(pinned);
    require(!p.start_demote(*one_tail), "demotion ignored a new KV reader");
    p.directory.pool(0).pages().unpin_source(pinned);
    auto altered = *one_tail;
    ++altered.host_bytes;
    require(!p.start_demote(altered), "demotion accepted a changed physical quote");

    const auto settle = [&] {
        require(p.operation && p.operation->ticket &&
                    std::holds_alternative<q4::detail::DemoteOperation>(p.operation->action),
                "unexpected native transaction in demotion fixture");
        p.operation->result.emplace(retire_context(executor, *p.operation->ticket));
        const auto completed = p.complete_context();
        require(completed.complete && completed.published && !p.operation,
                "demotion did not publish after physical retirement");
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
        auto device = std::make_unique<DeviceContext>(0);
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
        auto definition = q4::read_definition(reader, models::load_options(options));
        auto profile    = q4::execution::describe_model(*definition);
        auto load       = q4::resolve_load(reader, definition, device->device(), Device::cpu());
        auto execution  = q4::plan_execution(profile, load.ple_table().geometry(), options, *device,
                                             8ULL * 1024 * 1024 * 1024,
                                             q4::describe_moe_plan(*profile, options.moe));
        require(execution.has_value(), "reclaim test execution plan does not fit");
        auto session_plan = q4::plan_session(*execution);
        auto model        = q4::materialize_model(std::move(load), *device);
        auto instance     = std::make_unique<runtime::RankModelInstance<q4::RuntimeTypes>>(
            std::move(model), profile);
        instance->program =
            std::make_unique<q4::RankProgram>(instance->parameters, std::move(*execution), *device);
        auto worker = std::make_unique<runtime::RankWorker<q4::RuntimeTypes>>(std::move(device),
                                                                              std::move(instance));
        Executor executor(std::move(worker), {1});
        q4::detail::SessionImpl session(definition, std::move(session_plan));
        session.executor            = &executor;
        session.identity_generation = executor.epoch().value;
        reclaim(session, executor);
        std::cout << "OK Qwen4Exp recovery quotes, physical coverage and migrations\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
