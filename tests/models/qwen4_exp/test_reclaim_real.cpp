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
    const auto ticket = executor.submit(std::move(*reservation),
                                        {{{0, transaction.actions(0)}}, q4::ContextWork{}});
    transaction.mark_submitted();
    try {
        (void)retire_context(executor, ticket);
        transaction.mark_completed();
        transaction.publish();
    } catch (...) {
        try {
            executor.drain();
        } catch (...) {}
        if (transaction.in_flight()) transaction.fail_after_drain();
        transaction.discard();
        throw;
    }
}

void reclaim(q4::detail::SessionImpl& p, Executor& executor) {
    const auto live_bytes = [&](ContextMemoryDomain domain) {
        std::size_t bytes = 0;
        for (const auto& pool : p.usage().pools) {
            if (pool.domain == domain) bytes += pool.live_bytes;
        }
        return bytes;
    };
    const auto empty_device = live_bytes(ContextMemoryDomain::Device);
    const auto empty_host   = live_bytes(ContextMemoryDomain::Host);
    std::array<context::HistoryOwner, 2> owners;
    {
        auto bindings = p.directory.begin_transaction();
        context::ContextDemand demand;
        for (std::uint16_t pool = 0; pool < 2; ++pool) {
            demand.add_history(
                pool, p.directory.pool(pool).histories().binding_demand(nullptr, 0, 256U - pool));
        }
        auto reserved = p.directory.reserve(demand);
        require(reserved && reserved.reservation->ready(), "history reservation failed");
        for (std::uint16_t pool = 0; pool < 2; ++pool) {
            bindings.add_binding(
                pool, p.directory.pool(pool).histories().prepare_binding(
                          nullptr, 0, 256U - pool, 0, reserved.reservation->take_history(pool)));
        }
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
    const context::ContextDemand demand{.new_states = 1};
    const auto plan = p.plan_reclaim(all, {}, demand);
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
    require(tail->released.records.size() < tail_release->released.records.size(),
            "partial physical move was confused with deletion of its checkpoint");
    auto selection = plan.inventory->select();
    {
        auto probe = selection.probe({&shallow, 1});
        require(selection.released() ==
                    context::ContextReleaseSet{.catalog = {p.checkpoints.retirement(shallow)}},
                "retiring an alias must release only its catalog entry, not shared State/KV");
    }
    {
        auto probe = selection.probe(all);
        require(selection.released() == p.footprint(all, true),
                "co-holder inventory differs from actual complete release proof");
    }
    const auto extra = *p.checkpoints.reserve();
    p.checkpoints.publish(extra, p.checkpoints.at(deep));
    require(p.start_demote(*tail, {}) != runtime::ContextStart::Started,
            "stale quote ignored a newly published physical holder");
    require(p.checkpoints.erase(extra), "extra holder retirement failed");
    const auto& pinned_pages =
        tail->impl->pages[0].empty() ? tail->impl->pages[1] : tail->impl->pages[0];
    require(!pinned_pages.empty(), "suffix quote has no KV record");
    const auto pinned = pinned_pages.front();
    p.directory.pool(pinned.pool_id).pages().pin_source(pinned);
    require((p.start_demote(*tail, {}) != runtime::ContextStart::Started),
            "demotion ignored a new KV reader");
    p.directory.pool(pinned.pool_id).pages().unpin_source(pinned);
    auto altered = *tail;
    ++altered.host_bytes;
    require((p.start_demote(altered, {}) != runtime::ContextStart::Started),
            "demotion accepted a changed physical quote");

    const auto settle = [&] {
        while (p.operation) {
            require(p.operation->ticket &&
                        std::holds_alternative<q4::detail::DemoteOperation>(p.operation->action),
                    "unexpected native transaction in demotion fixture");
            p.operation->result.emplace(retire_context(executor, *p.operation->ticket));
            const auto completed = p.complete_context();
            require(!completed.complete || (completed.published && !p.operation),
                    "demotion did not publish after physical retirement");
        }
    };
    auto current = p.plan_reclaim(all, {}, demand);
    require(!current.demotions.empty(), "demotion quote missing");
    const auto quote           = current.demotions.front();
    const auto device_before   = live_bytes(ContextMemoryDomain::Device);
    const auto host_before     = live_bytes(ContextMemoryDomain::Host);
    const auto released_device = p.directory.released_bytes(quote.released);
    require(p.start_demote(quote, {}) == runtime::ContextStart::Started,
            "demotion refused a valid physical quote");
    settle();
    require(live_bytes(ContextMemoryDomain::Device) + released_device == device_before &&
                live_bytes(ContextMemoryDomain::Host) == host_before + quote.host_bytes,
            "demotion did not account for its actual Device and Host records");
    require(p.start_demote(quote, {}) != runtime::ContextStart::Started,
            "stale residency quote was reused");

    const auto remaining = p.demotion(all);
    require(remaining && !remaining->images.empty(), "State demotion missing");
    const auto whole = p.describe_demotion(remaining);
    require(p.start_demote(whole, {}) == runtime::ContextStart::Started,
            "complete State/Main/MTP demotion refused its reserved destination");
    settle();
    auto host = p.plan_releases(all, {}, {.new_states = 1, .memory = context::ContextMemory::Host});
    auto selected     = host.inventory->select();
    auto probe        = selected.probe(all);
    auto host_records = selected.released();
    std::erase_if(host_records.records, [](const auto& record) {
        return record.domain.memory != context::ContextMemory::Host;
    });
    const auto quoted_host   = p.directory.released_bytes(host_records);
    const auto occupied_host = live_bytes(ContextMemoryDomain::Host);
    p.checkpoints.clear();
    require(occupied_host - live_bytes(ContextMemoryDomain::Host) == quoted_host &&
                live_bytes(ContextMemoryDomain::Device) == empty_device &&
                live_bytes(ContextMemoryDomain::Host) == empty_host,
            "physical release proof differed from actual Native occupancy");
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
        options.use_cuda_graph   = false;
        options.enable_vision    = false;
        options.max_context      = 512;
        options.prefill_chunk    = 256;
        options.max_concurrency  = 1;
        options.context_capacity = ContextCapacityPolicy::explicit_capacity(1024ULL * 1024 * 1024);
        options.speculative.backend               = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens          = 1;
        // Two State records and the Main/MTP record classes need four Context blocks.
        options.context_cache.host_capacity_bytes = 512ULL * 1024 * 1024;
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
