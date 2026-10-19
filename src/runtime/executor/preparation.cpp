#include "runtime/executor/preparation.h"

#include <algorithm>
#include <thread>

namespace ninfer::runtime {
namespace {
namespace dist = distributed;
namespace wire = dist::wire;

void broadcast(dist::ExecutionGroup& group, wire::Kind kind, const wire::Bytes& payload) {
    const auto message =
        std::make_shared<const wire::Message>(wire::Message{kind, group.epoch(), payload});
    std::vector<dist::OutgoingMessage> messages;
    for (const auto& rank : group.topology().ranks) messages.push_back({rank.rank, message});
    if (!group.transport().try_send(messages, distributed::Traffic::Continuation))
        throw std::runtime_error("preparation control queue is full");
}

std::vector<wire::Bytes> receive(dist::ExecutionGroup& group, wire::Kind kind,
                                 std::chrono::milliseconds timeout) {
    std::vector<wire::Bytes> result(group.topology().ranks.size());
    std::vector<bool> received(result.size());
    std::size_t remaining = result.size();
    const auto deadline   = std::chrono::steady_clock::now() + timeout;
    while (remaining) {
        group.check_processes();
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("rank model preparation timed out");
        auto message = group.transport().poll();
        if (!message) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (message->message.epoch != group.epoch())
            throw std::runtime_error("stale preparation epoch");
        if (message->message.kind == wire::Kind::Failed) {
            wire::Decoder in(message->message.payload);
            throw std::runtime_error("rank " + std::to_string(message->peer) + ": " + in.string());
        }
        if (message->message.kind != kind || received.at(message->peer))
            throw std::runtime_error("unexpected rank preparation phase");
        received[message->peer] = true;
        result[message->peer]   = std::move(message->message.payload);
        --remaining;
    }
    return result;
}
} // namespace

void encode_preparation_report(wire::Encoder& out, const RankPreparationReport& r) {
    const auto& c = r.capacity.curve;
    out.u32(c.primary_page_tokens);
    out.u32(c.minimum_primary_page_groups);
    out.u32(c.maximum_primary_page_groups);
    out.u64(c.minimum_device_reservation_bytes);
    out.u64(c.bytes_per_additional_primary_page_group);
    out.u64(r.capacity.available_bytes);
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
    out.string(r.hardware_class);
    out.string(r.cuda_sync_mode);
}

RankPreparationReport decode_preparation_report(wire::Decoder& in) {
    RankPreparationReport r;
    r.capacity.curve           = {in.u32(), in.u32(), in.u32(), in.u64(), in.u64()};
    r.capacity.available_bytes = in.u64();
    r.weight_bytes             = in.u64();
    r.host_weight_bytes        = in.u64();
    r.staging_bytes            = in.u64();
    r.read_bytes               = in.u64();
    r.upload_bytes             = in.u64();
    r.load_seconds             = in.f64();
    r.device_objects           = in.u64();
    r.host_objects             = in.u64();
    r.free_after_weights_bytes = in.u64();
    r.preparation_required     = in.boolean();
    r.hardware_class           = in.string();
    r.cuda_sync_mode           = in.string();
    return r;
}

GroupPreparation prepare_execution_group(dist::ExecutionGroup& group,
                                         std::span<const wire::Bytes> payloads,
                                         const KvCapacityPolicy& policy,
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
        if (!group.transport().try_send(requests))
            throw std::runtime_error("rank preparation payloads exceed control capacity");
        GroupPreparation result;
        for (std::uint32_t attempt = 0; attempt < 2; ++attempt) {
            const auto enter = [&](PreparationAction action) {
                wire::Encoder out;
                out.u64(group.topology().plan);
                out.u32(attempt);
                out.u8(static_cast<std::uint8_t>(action));
                out.u32(result.pages);
                broadcast(group, wire::Kind::Enter, std::move(out).take());
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
                for (const auto& bytes : receive(group, wire::Kind::Allocated, timeout)) {
                    wire::Decoder in(bytes);
                    if (in.u64() != group.topology().plan || in.u32() != attempt)
                        throw std::runtime_error(
                            "rank preparation response belongs to another attempt");
                    success &= in.boolean();
                    if (description) {
                        const auto layout = in.bytes();
                        result.layouts.emplace_back(layout.begin(), layout.end());
                    }
                    in.finish();
                }
                return success;
            };
            reports();
            if (attempt == 0 &&
                std::any_of(result.reports.begin(), result.reports.end(),
                            [](const auto& report) { return report.preparation_required; })) {
                enter(PreparationAction::PrepareProbe);
                if (!allocated(false))
                    throw DeviceAllocationError("rank calibration resources do not fit");
                enter(PreparationAction::Calibrate);
                reports();
            }
            std::vector<RankCapacityOffer> offers;
            for (const auto& report : result.reports) offers.push_back(report.capacity);
            const auto capacities = resolve_group_kv_capacity(policy, offers);
            result.pages          = capacities.front().primary_page_groups;
            enter(PreparationAction::Allocate);
            bool success = allocated(true);
            if (success) {
                enter(PreparationAction::PrepareExecution);
                success = allocated(true);
            }
            if (success) {
                if (control.prepare) control.prepare(result.layouts);
                enter(PreparationAction::CheckMemory);
                if (allocated(true)) {
                    result.attempt = attempt;
                    return result;
                }
            }
            if (control.release) control.release();
            if (attempt == 1)
                throw std::runtime_error("rank resources do not fit the common capacity");
            enter(PreparationAction::Release);
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
        wire::Encoder out;
        out.u64(group.topology().plan);
        out.u32(prepared.attempt);
        out.u8(static_cast<std::uint8_t>(PreparationAction::Activate));
        out.u32(prepared.pages);
        broadcast(group, wire::Kind::Enter, std::move(out).take());
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
