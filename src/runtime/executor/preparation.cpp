#include "runtime/executor/preparation.h"

#include <algorithm>

namespace ninfer::runtime {
namespace {
namespace dist = distributed;
namespace wire = dist::wire;

void enter(dist::ExecutionGroup& group, const GroupPreparation& prepared, std::uint32_t attempt,
           PreparationAction action) {
    std::vector<dist::OutgoingMessage> messages;
    for (const auto& rank : group.topology().ranks) {
        const auto* capacity =
            prepared.capacities.empty() ? nullptr : &prepared.capacities.at(rank.rank);
        wire::Encoder out;
        out.u64(group.topology().plan);
        out.u32(attempt);
        out.u8(static_cast<std::uint8_t>(action));
        out.u64(capacity ? capacity->device_bytes : 0);
        out.u64(capacity ? capacity->control_reservation_bytes : 0);
        out.u64(capacity ? capacity->planned_slack_bytes : 0);
        messages.push_back(
            {rank.rank, std::make_shared<const wire::Message>(wire::Message{
                            wire::Kind::Enter, group.epoch(), std::move(out).take()})});
    }
    if (!group.try_send(messages, distributed::Traffic::Continuation))
        throw std::runtime_error("preparation control queue is full");
}

std::vector<wire::Payload> receive(dist::ExecutionGroup& group, wire::Kind kind,
                                   std::chrono::milliseconds timeout) {
    std::vector<wire::Payload> result(group.topology().ranks.size());
    for (auto& message : group.receive_all(kind, timeout))
        result.at(message.peer) = std::move(message.message.payload);
    return result;
}
} // namespace

void encode_preparation_report(wire::Encoder& out, const RankPreparationReport& r) {
    const auto& plan = r.capacity.plan;
    out.u64(plan.fixed_device_bytes);
    out.u64(plan.fixed_host_bytes);
    out.u32(plan.records.size());
    for (const auto& record : plan.records) {
        out.u64(record.bytes);
        out.u64(record.alignment);
    }
    out.u32(plan.working_sets.size());
    for (const auto& set : plan.working_sets) {
        out.string(set.name);
        out.u32(set.records.size());
        for (const auto& record : set.records) {
            out.u32(record.record_class.index);
            out.u32(record.count);
        }
    }
    out.u64(plan.metadata.fixed_bytes);
    out.u64(plan.metadata.bytes_per_block);
    out.u32(plan.metadata.bytes_per_record.size());
    for (const auto bytes : plan.metadata.bytes_per_record) out.u64(bytes);
    out.u32(plan.reference.tokens_per_request);
    out.u32(plan.reference.request_count);
    out.u64(plan.reference.device_bytes);
    out.u64(r.capacity.available_bytes);
    out.u64(r.capacity.available_host_bytes);
    out.boolean(r.capacity.shared_physical_memory);
    out.u64(r.weight_bytes);
    out.u64(r.host_weight_bytes);
    out.u64(r.staging_bytes);
    out.u64(r.read_bytes);
    out.u64(r.upload_bytes);
    out.f64(r.load_seconds);
    out.u64(r.device_objects);
    out.u64(r.host_objects);
    out.u64(r.free_after_weights_bytes);
    out.boolean(r.preparation_required);
    out.boolean(r.control_owner);
    out.string(r.hardware_class);
    out.string(r.cuda_sync_mode);
}

RankPreparationReport decode_preparation_report(wire::Decoder& in) {
    RankPreparationReport r;
    auto& plan              = r.capacity.plan;
    plan.fixed_device_bytes = in.u64();
    plan.fixed_host_bytes   = in.u64();
    plan.records.resize(in.u32());
    for (auto& record : plan.records) record = {in.u64(), in.u64()};
    plan.working_sets.resize(in.u32());
    for (auto& set : plan.working_sets) {
        set.name = in.string();
        set.records.resize(in.u32());
        for (auto& record : set.records) record = {{in.u32()}, in.u32()};
    }
    plan.metadata.fixed_bytes     = in.u64();
    plan.metadata.bytes_per_block = in.u64();
    plan.metadata.bytes_per_record.resize(in.u32());
    for (auto& bytes : plan.metadata.bytes_per_record) bytes = in.u64();
    plan.reference                    = {in.u32(), in.u32(), in.u64()};
    r.capacity.available_bytes        = in.u64();
    r.capacity.available_host_bytes   = in.u64();
    r.capacity.shared_physical_memory = in.boolean();
    r.weight_bytes                    = in.u64();
    r.host_weight_bytes               = in.u64();
    r.staging_bytes                   = in.u64();
    r.read_bytes                      = in.u64();
    r.upload_bytes                    = in.u64();
    r.load_seconds                    = in.f64();
    r.device_objects                  = in.u64();
    r.host_objects                    = in.u64();
    r.free_after_weights_bytes        = in.u64();
    r.preparation_required            = in.boolean();
    r.control_owner                   = in.boolean();
    r.hardware_class                  = in.string();
    r.cuda_sync_mode                  = in.string();
    return r;
}

GroupPreparation prepare_execution_group(dist::ExecutionGroup& group,
                                         std::span<const wire::Bytes> payloads,
                                         ContextCapacityPolicy policy,
                                         std::chrono::milliseconds timeout,
                                         ControlPreparation control) {
    if (payloads.size() != group.topology().ranks.size())
        throw std::invalid_argument("preparation payload count differs from rank count");
    try {
        std::vector<dist::OutgoingMessage> requests;
        for (const auto& rank : group.topology().ranks) {
            wire::Encoder out;
            out.u64(group.topology().plan);
            out.bytes(payloads[rank.rank]);
            requests.push_back(
                {rank.rank, std::make_shared<const wire::Message>(wire::Message{
                                wire::Kind::Prepare, group.epoch(), std::move(out).take()})});
        }
        if (!group.try_send(requests))
            throw std::runtime_error("rank preparation payloads exceed control capacity");
        GroupPreparation result;
        std::string allocation_failure;
        for (std::uint32_t attempt = 0; attempt < 2; ++attempt) {
            const auto enter_phase = [&](PreparationAction action) {
                enter(group, result, attempt, action);
            };
            const auto reports = [&] {
                result.reports.clear();
                for (const auto& bytes : receive(group, wire::Kind::Prepared, timeout)) {
                    wire::Decoder in(bytes);
                    if (in.u64() != group.topology().plan || in.u32() != attempt)
                        throw std::runtime_error("rank capacity belongs to another attempt");
                    result.reports.push_back(decode_preparation_report(in));
                    in.finish();
                }
            };
            const auto allocated = [&](bool description) {
                bool success = true;
                if (description) result.layouts.clear();
                const auto replies = receive(group, wire::Kind::Allocated, timeout);
                for (std::size_t rank = 0; rank < replies.size(); ++rank) {
                    const auto& bytes = replies[rank];
                    wire::Decoder in(bytes);
                    if (in.u64() != group.topology().plan || in.u32() != attempt)
                        throw std::runtime_error(
                            "rank preparation response belongs to another attempt");
                    const bool ready = in.boolean();
                    success &= ready;
                    const auto failure = in.string();
                    if (!ready)
                        allocation_failure = "rank " + std::to_string(rank) + ": " + failure;
                    if (description) {
                        const auto layout = in.bytes();
                        result.layouts.emplace_back(wire::Bytes(layout.begin(), layout.end()));
                    }
                    in.finish();
                }
                return success;
            };
            reports();
            if (attempt == 0) {
                enter_phase(PreparationAction::PrepareCommunication);
                reports();
            }
            if (attempt == 0 &&
                std::any_of(result.reports.begin(), result.reports.end(),
                            [](const auto& report) { return report.preparation_required; })) {
                enter_phase(PreparationAction::PrepareProbe);
                if (!allocated(false)) throw DeviceAllocationError(allocation_failure.c_str());
                enter_phase(PreparationAction::Calibrate);
                reports();
                enter_phase(PreparationAction::PrepareRebuildProbe);
                if (!allocated(false)) throw DeviceAllocationError(allocation_failure.c_str());
                enter_phase(PreparationAction::CalibrateRebuild);
                reports();
            }
            std::vector<RankCapacityOffer> offers;
            for (const auto& report : result.reports) offers.push_back(report.capacity);
            std::optional<std::size_t> control_owner;
            for (std::size_t rank = 0; rank < result.reports.size(); ++rank) {
                const auto& report = result.reports[rank];
                if (report.control_owner) {
                    if (control_owner)
                        throw std::logic_error("Context directory has multiple memory owners");
                    control_owner = rank;
                }
            }
            if (!control_owner) throw std::logic_error("Context directory memory owner is absent");
            result.capacities = resolve_group_context_capacity(policy, offers, *control_owner);
            enter_phase(PreparationAction::Allocate);
            bool success = allocated(true);
            if (success) {
                enter_phase(PreparationAction::PrepareExecution);
                success = allocated(true);
            }
            if (success) {
                if (control.prepare) control.prepare(result.layouts);
                enter_phase(PreparationAction::CheckMemory);
                if (allocated(true)) {
                    result.attempt = attempt;
                    return result;
                }
            }
            if (attempt == 1 || policy.mode == ContextCapacityMode::Explicit)
                throw std::runtime_error(allocation_failure);
            if (control.release) control.release();
            enter_phase(PreparationAction::Release);
        }
        throw std::logic_error("preparation attempt exhausted");
    } catch (const std::exception& error) {
        if (control.release) control.release();
        group.fail(error.what());
        throw;
    }
}

void activate_execution_group(dist::ExecutionGroup& group, const GroupPreparation& prepared,
                              std::chrono::milliseconds timeout) {
    try {
        enter(group, prepared, prepared.attempt, PreparationAction::Activate);
        for (const auto& bytes : receive(group, wire::Kind::Ready, timeout)) {
            wire::Decoder in(bytes);
            if (in.u64() != group.topology().plan || in.u32() != prepared.attempt)
                throw std::runtime_error("rank readiness belongs to another preparation attempt");
            in.finish();
        }
    } catch (const std::exception& error) {
        group.fail(error.what());
        throw;
    }
}
} // namespace ninfer::runtime
