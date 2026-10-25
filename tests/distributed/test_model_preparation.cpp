#include "models/qwen4_exp/parallel/codec.h"
#include "models/qwen4_exp/startup.h"
#include "product/config/options.h"
#include "runtime/executor/distributed_executor.h"
#include "runtime/executor/preparation.h"

#include <iostream>
#include <thread>

namespace {
using namespace ninfer;
namespace q4   = models::qwen4_exp;
namespace rt   = runtime;
namespace wire = distributed::wire;
using Executor = rt::DistributedExecutor<q4::ExecutionTypes, q4::ExecutionCodec>;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

std::vector<wire::Bytes> exchange(distributed::ExecutionGroup& group,
                                  const rt::GroupPreparation& prepared,
                                  rt::PreparationAction action, wire::Kind reply) {
    std::vector<distributed::OutgoingMessage> messages;
    for (const auto& rank : group.topology().ranks) {
        const auto& capacity = prepared.capacities.at(rank.rank);
        wire::Encoder out;
        out.u64(group.topology().plan);
        out.u32(prepared.attempt);
        out.u8(static_cast<std::uint8_t>(action));
        out.u64(capacity.device_bytes);
        out.u64(capacity.control_reservation_bytes);
        messages.push_back(
            {rank.rank, std::make_shared<const wire::Message>(wire::Message{
                            wire::Kind::Enter, group.epoch(), std::move(out).take()})});
    }
    require(group.transport().try_send(messages, distributed::Traffic::Continuation),
            "preparation queue is full");
    std::vector<wire::Bytes> responses(messages.size());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(2);
    for (std::size_t n = 0; n < responses.size();) {
        group.check_processes();
        require(std::chrono::steady_clock::now() < deadline, "preparation response timed out");
        if (auto incoming = group.transport().poll()) {
            if (incoming->message.kind == wire::Kind::Failed) {
                wire::Decoder in(incoming->message.payload);
                throw std::runtime_error(in.string());
            }
            require(incoming->message.epoch == group.epoch() && incoming->message.kind == reply &&
                        responses.at(incoming->peer).empty(),
                    "unexpected preparation response");
            responses[incoming->peer] = std::move(incoming->message.payload);
            ++n;
        } else
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return responses;
}

void rebuild(distributed::ExecutionGroup& group, rt::GroupPreparation& prepared) {
    auto reports = exchange(group, prepared, rt::PreparationAction::Release, wire::Kind::Prepared);
    ++prepared.attempt;
    for (std::size_t i = 0; i < reports.size(); ++i) {
        wire::Decoder in(reports[i]);
        require(in.u64() == group.topology().plan && in.u32() == prepared.attempt,
                "rebuild report has the wrong attempt");
        const auto report = rt::decode_preparation_report(in);
        in.finish();
        require(report.read_bytes == prepared.reports[i].read_bytes &&
                    report.upload_bytes == prepared.reports[i].upload_bytes &&
                    report.weight_bytes == prepared.reports[i].weight_bytes,
                "Program rebuild reloaded immutable weights");
        prepared.reports[i] = report;
    }
    prepared.layouts.clear();
    for (const auto& bytes :
         exchange(group, prepared, rt::PreparationAction::Allocate, wire::Kind::Allocated)) {
        wire::Decoder in(bytes);
        require(in.u64() == group.topology().plan && in.u32() == prepared.attempt && in.boolean(),
                "Program rebuild did not fit");
        const auto layout = in.bytes();
        prepared.layouts.emplace_back(layout.begin(), layout.end());
        in.finish();
    }
}

q4::ContextResult execute(Executor& executor, q4::Work work) {
    auto reservation = executor.try_reserve(rt::FrameKind::Context);
    require(reservation.has_value(), "context frame was not retired");
    const auto ticket = executor.submit(std::move(*reservation), std::move(work));
    std::optional<q4::ContextResult> result;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
        if (auto event = executor.poll()) {
            if (const auto* error = std::get_if<rt::ExecutionFailed>(&*event))
                throw std::runtime_error(error->message);
            require(std::visit([&](const auto& value) { return value.ticket == ticket; }, *event),
                    "context event belongs to another execution");
            if (auto* ready = std::get_if<rt::ResultReady<q4::ExecutionTypes>>(&*event)) {
                require(!result, "duplicate context result");
                result = std::get<q4::ContextResult>(std::move(ready->result));
            } else if (std::holds_alternative<rt::DeviceRetired>(*event)) {
                require(result.has_value(), "context retired before all owners completed");
                return std::move(*result);
            } else
                throw std::runtime_error("unexpected context event");
        } else
            std::this_thread::yield();
    }
    throw std::runtime_error("context execution timed out");
}

void contexts(Executor& executor, const q4::SessionPlan& plan) {
    namespace ctx = models::context;
    std::vector<ctx::ContextRecordLayout> layouts;
    for (const auto& state : plan.context.state.layouts) {
        ctx::ContextRecordLayout layout{static_cast<std::uint32_t>(layouts.size()), {}};
        for (const auto& part : state.parts)
            layout.parts.push_back({part.rank, {part.image_bytes, part.alignment}});
        layouts.push_back(std::move(layout));
    }
    const auto state_count = layouts.size();
    for (const auto& pool : plan.context.pools)
        layouts.push_back({static_cast<std::uint32_t>(layouts.size()), pool.parts});
    ctx::RecordDomains records(plan.context.ranks, layouts);
    std::vector<ctx::RecordDemand> demands;
    for (const auto& layout : layouts)
        demands.push_back({layout.id, ctx::ContextMemory::Device, 2, {}});
    for (std::uint64_t generation = 1; generation <= 2; ++generation) {
        auto reserved = records.reserve(demands);
        require(reserved.reservations.has_value() && reserved.assessment.ready(),
                "context fixture records do not fit");
        std::vector<ctx::RecordLease> retained;
        const rt::ExecutionMember member{{generation, 1}, {{0}, generation}};
        q4::ContextWork binding;
        binding.bindings.push_back({member, {}, {1, 2, 1}});
        q4::Work work{{}, std::move(binding)};
        std::uint32_t expected_images = 0, expected_pages = 0;
        for (std::size_t i = 0; i < layouts.size(); ++i) {
            auto source      = reserved.reservations->at(i).materialize();
            auto destination = reserved.reservations->at(i).materialize();
            for (std::size_t part = 0; part < source.size(); ++part) {
                const auto from = source[part].binding();
                const auto to   = destination[part].binding();
                require(from.domain == to.domain, "fixture record placement differs");
                auto& actions = work.actions[from.domain.rank].operations;
                const ctx::ContextRange input{from.domain.memory, from.offset, from.bytes};
                const ctx::ContextRange output{to.domain.memory, to.offset, to.bytes};
                actions.emplace_back(ctx::RecordClear{input});
                const rt::ContextTransferResource resource{
                    i < state_count
                        ? std::nullopt
                        : std::optional{rt::KvPoolId{static_cast<std::uint16_t>(i - state_count)}}};
                actions.emplace_back(ctx::RecordCopy{
                    input, output, resource, i < state_count ? 1U : 0U, i < state_count ? 0U : 1U});
                if (i < state_count)
                    ++expected_images;
                else {
                    ++expected_pages;
                    actions.emplace_back(
                        ctx::MappingUpdate{static_cast<std::uint32_t>(i - state_count),
                                           0,
                                           0,
                                           {from.offset, to.offset}});
                }
                retained.push_back(std::move(source[part]));
                retained.push_back(std::move(destination[part]));
            }
        }
        const auto result    = execute(executor, std::move(work));
        std::uint32_t images = 0, pages = 0;
        for (const auto& transfer : result.transfers) {
            images += transfer.image_count;
            pages += transfer.page_groups;
            require(transfer.work.payload_bytes && transfer.work.copy_operations,
                    "context copy did not report physical work");
        }
        require(images == expected_images && pages == expected_pages,
                "context completion omitted rank-local physical copies");
        (void)execute(executor, {{}, q4::ContextWork{{}, {member.residency}}});
    }
}

void bind_requests(q4::Session& session, models::qwen::Frontend& frontend) {
    for (std::uint32_t lane : {0U, 1U, 0U}) {
        auto request      = session.plan_request(frontend.prepare_tokens({1, 2, 3, 4}),
                                                 {.requested_output_tokens = 8});
        const auto source = session.inspect_source(request, {});
        require(source.has_value(), "fresh context has no source plan");
        require(bool(session.start_binding(request, {lane}, *source, nullptr,
                                           q4::ExecutionUnitKind::Prefill, 4)),
                "Session could not bind a fresh context");
        std::optional<q4::SequenceHandle> sequence;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!sequence) {
            require(std::chrono::steady_clock::now() < deadline, "Session binding timed out");
            if (auto event = session.poll()) {
                const auto* result = std::get_if<q4::ContextProgress>(&event->result);
                require(result && result->complete && result->published && result->sequence,
                        "Session published an incomplete context");
                sequence = result->sequence;
            } else
                std::this_thread::yield();
        }
        require(session.abort(*sequence).status == rt::ConsumeStatus::Consumed,
                "Session did not release its bound context");
        while (session.has_context_operation()) {
            require(std::chrono::steady_clock::now() < deadline, "Session release timed out");
            (void)session.poll();
        }
        require(session.physical_usage().state_objects == 0, "Session retained an aborted State");
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3) return 77;
    try {
        auto deployment = product::read_deployment_file(argv[1]);
        artifact::Reader reader(argv[2]);
        EngineOptions options;
        options.use_cuda_graph   = false;
        options.enable_vision    = true;
        options.max_context      = 4096;
        options.max_concurrency  = 2;
        options.prefill_chunk    = 4096;
        options.context_capacity = ContextCapacityPolicy::explicit_capacity(1ULL << 30);
        options.context_cache.host_capacity_bytes = 0;
        options.speculative = {.backend = SpeculativeBackend::Mtp, .draft_tokens = 3};
        auto definition     = q4::read_definition(reader, models::load_options(options));
        auto frontend       = q4::create_frontend(*definition, options);
        const auto parallel = q4::plan_parallel(*definition, deployment);
        std::vector<wire::Bytes> payloads;
        for (const auto& rank : parallel.ranks) {
            wire::Encoder out;
            const auto node = parallel.topology.rank(rank.rank).node;
            q4::encode_rank_load(
                out, {definition->artifact_id(), rank, options,
                      node == deployment.coordinator_node ? options.media_live_bytes : 0,
                      node == deployment.coordinator_node});
            payloads.push_back(std::move(out).take());
        }
        auto group = std::make_unique<distributed::ExecutionGroup>(
            deployment, parallel.topology, [](auto rank, std::string_view text) {
                std::cerr << "[rank " << rank << "] " << text << '\n';
            });
        auto prepared = rt::prepare_execution_group(*group, payloads, options.context_capacity,
                                                    std::chrono::minutes(15));
        rebuild(*group, prepared);
        std::vector<q4::RankExecutionDescription> descriptions;
        for (std::size_t i = 0; i < prepared.layouts.size(); ++i) {
            wire::Decoder in(prepared.layouts[i]);
            auto d = q4::decode_rank_execution(in);
            in.finish();
            require(d.budget.context_capacity_bytes == prepared.capacities[i].device_bytes &&
                        d.capacity.compute_frames && d.capacity.context_frames &&
                        !prepared.reports[i].host_weight_bytes,
                    "prepared execution resources differ from the native plan");
            std::cout << "rank=" << i << " weights=" << prepared.reports[i].weight_bytes
                      << " runtime=" << d.budget.runtime_reservation_bytes
                      << " context=" << prepared.capacities[i].device_bytes
                      << " attempt=" << prepared.attempt
                      << " read=" << prepared.reports[i].read_bytes
                      << " upload=" << prepared.reports[i].upload_bytes << std::endl;
            descriptions.push_back(std::move(d));
        }
        const auto plan = q4::plan_group_execution(parallel, descriptions);
        q4::ExecutionCodec codec(*definition, descriptions.front(), parallel.topology);
        rt::activate_execution_group(*group, prepared, std::chrono::seconds(30));
        std::vector<RankSnapshot> initial;
        for (const auto& d : descriptions) initial.push_back(d.observation);
        Executor executor(std::move(group), plan.capacity, std::move(codec), std::move(initial));
        executor.bind_to_current_thread();
        contexts(executor, plan.session);
        require(executor.observations().size() == descriptions.size(),
                "per-rank resource observations were lost");
        for (std::size_t i = 0; i < descriptions.size(); ++i)
            require(executor.observations()[i].rank == i &&
                        executor.observations()[i].memory.allocations.device_weights_bytes ==
                            descriptions[i].observation.memory.allocations.device_weights_bytes,
                    "resource observation confused rank identities or backing ownership");
        {
            q4::Session running(q4::prepare_session(definition, plan.session), executor);
            bind_requests(running, frontend);
        }
        require(!executor.failed() && !executor.has_pending_work(), "execution did not drain");
        std::cout << "PASS: native Program rebuild, activation, context operations, Session "
                     "binding and shutdown\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
