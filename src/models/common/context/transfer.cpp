#include "models/common/context/transfer.h"

#include "distributed/communication.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <exception>
#include <string>
#include <limits>
#include <optional>
#include <stdexcept>

namespace ninfer::models::context {
namespace {
std::size_t mapping_bytes(const ContextStorage& storage, std::size_t requested) {
    const auto elements = requested ? requested : storage.mapping_capacity_elements();
    if (elements > SIZE_MAX / sizeof(std::int64_t))
        throw std::overflow_error("context mapping staging size overflow");
    return elements * sizeof(std::int64_t);
}
} // namespace

ContextMigration::ContextMigration(DeviceContext& device, ContextStorage& storage,
                                   std::size_t mapping_capacity_elements,
                                   std::size_t copy_segment_capacity)
    : ContextMigration(device, storage, device.transfer_view(), mapping_capacity_elements,
                       copy_segment_capacity) {}

ContextMigration::ContextMigration(DeviceContext& device, ContextStorage& storage,
                                   CudaExecutionView execution,
                                   std::size_t mapping_capacity_elements,
                                   std::size_t copy_segment_capacity, ContextPeerBinding peer)
    : device_(&device), storage_(&storage), execution_(execution), peer_group_(peer.group),
      peer_members_(peer.members.begin(), peer.members.end()), peer_staging_(peer.staging),
      peer_chunk_bytes_(peer.chunk_bytes),
      mapping_(mapping_bytes(storage, mapping_capacity_elements)), copies_(device.device(), 64),
      source_ready_(device), completed_(device) {
    if (execution.device() != device.device())
        throw std::invalid_argument("context action stream belongs to another device");
    if (peer_group_) {
#if NINFER_ENABLE_DISTRIBUTED
        validate_memory(peer_staging_);
        if (peer_group_->device() != device.device() ||
            peer_members_.size() != peer_group_->size() ||
            peer_members_[peer_group_->rank()] != storage.logical_rank() ||
            peer_staging_.device != device.device() || peer_chunk_bytes_ == 0 ||
            peer_staging_.bytes < peer_chunk_bytes_)
            throw std::invalid_argument("Context peer binding has inconsistent startup resources");
        for (std::size_t i = 0; i < peer_members_.size(); ++i)
            if (std::find(peer_members_.begin(), peer_members_.begin() + i, peer_members_[i]) !=
                peer_members_.begin() + i)
                throw std::invalid_argument("Context peer group contains a duplicate rank");
#else
        throw std::invalid_argument("Context peer transfer requires a distributed build");
#endif
    } else if (!peer_members_.empty() || peer_staging_.bytes || peer_chunk_bytes_) {
        throw std::invalid_argument("Context peer resources require their communication group");
    }
    const auto capacity =
        copy_segment_capacity ? copy_segment_capacity : storage.copy_segment_capacity();
    timers_.reserve(capacity);
    segments_.reserve(capacity);
    observations_.reserve(capacity);
    for (std::size_t i = 0; i < capacity; ++i) timers_.emplace_back(device, execution.stream());
}

ContextMigration::~ContextMigration() {
    if (!ready()) std::terminate();
}

void ContextMigration::validate(const ContextActions& actions) const {
    storage_->validate(actions, mapping_.span());
    std::size_t segments = 0;
    std::optional<runtime::ContextTransferObservation> previous;
    for (const auto& action : actions.operations) {
        if (const auto* peer = std::get_if<PeerCopy>(&action)) {
            if (!peer_group_ || std::find(peer_members_.begin(), peer_members_.end(), peer->peer) ==
                                    peer_members_.end())
                throw std::invalid_argument("Context peer is outside the prepared group");
            const auto staging = reinterpret_cast<std::uintptr_t>(peer_staging_.data);
            for (const auto& range : peer->ranges) {
                const auto local   = storage_->range(range);
                const auto address = reinterpret_cast<std::uintptr_t>(local.data);
                const bool overlap = address <= staging ? staging - address < local.bytes
                                                        : address - staging < peer_chunk_bytes_;
                if (overlap)
                    throw std::invalid_argument("Context peer staging overlaps its record ranges");
            }
        }
        const auto observed = storage_->describe(action);
        if (observed.peer && 1 + (observed.work.payload_bytes - 1) / peer_chunk_bytes_ > UINT32_MAX)
            throw std::overflow_error("Context peer chunk count exceeds its observation range");
        if (!observed.work.copy_operations) {
            previous.reset();
            continue;
        }
        if (!previous || previous->resource != observed.resource ||
            previous->direction != observed.direction || previous->peer != observed.peer)
            ++segments;
        previous = observed;
    }
    if (segments > timers_.size())
        throw std::out_of_range("context action segments exceed the prepared slot");
}

void ContextMigration::prepare(const ContextActions& actions) {
    if (occupied_) throw std::logic_error("context action slot is occupied");
    validate(actions);
    segments_.clear();
    observations_.clear();
    for (std::size_t i = 0; i < actions.operations.size(); ++i) {
        const auto observed = storage_->describe(actions.operations[i]);
        if (!observed.work.copy_operations) continue;
        if (segments_.empty() || segments_.back().end != i ||
            segments_.back().observed.resource != observed.resource ||
            segments_.back().observed.direction != observed.direction ||
            segments_.back().observed.peer != observed.peer) {
            if (segments_.size() == timers_.size())
                throw std::out_of_range("context action segments exceed the prepared slot");
            segments_.push_back({observed, i, i + 1});
        } else {
            auto& segment = segments_.back();
            segment.end   = i + 1;
            segment.observed.work.payload_bytes += observed.work.payload_bytes;
            segment.observed.work.copy_operations += observed.work.copy_operations;
            segment.observed.page_groups += observed.page_groups;
            segment.observed.image_count += observed.image_count;
        }
    }
}

void ContextMigration::enqueue(const ContextActions& actions, CudaExecutionView execution) {
    occupied_ = true;
    if (!storage_->uses_cuda(actions)) {
        submitted_stream_   = nullptr;
        fenced_             = false;
        std::size_t segment = 0;
        auto started        = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < actions.operations.size(); ++i) {
            if (segment < segments_.size() && segments_[segment].begin == i) {
                started = std::chrono::steady_clock::now();
            }
            storage_->execute(actions.operations[i], {}, execution);
            if (segment < segments_.size() && segments_[segment].end == i + 1) {
                segments_[segment].observed.elapsed_ns =
                    std::max<std::int64_t>(1, std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                  std::chrono::steady_clock::now() - started)
                                                  .count());
                ++segment;
            }
        }
        fenced_ = true;
        return;
    }
    fenced_           = false;
    submitted_stream_ = execution.stream();
    if (execution.stream() != device_->stream()) {
        source_ready_.record(device_->stream());
        source_ready_.wait(execution.stream());
    }
    auto staging        = mapping_.span();
    std::size_t segment = 0;
    copies_.clear();
    const auto flush = [&] {
        if (!copies_.size()) return;
        copies_.submit(execution);
        if (segment < segments_.size())
            segments_[segment].observed.work.copy_operations +=
                static_cast<std::uint32_t>(copies_.size());
        copies_.clear();
    };
    for (std::size_t i = 0; i < actions.operations.size(); ++i) {
        if (segment < segments_.size() && segments_[segment].begin == i) {
            segments_[segment].observed.work.copy_operations = 0;
            timers_[segment].start();
        }
        if (const auto* copy = std::get_if<RecordCopy>(&actions.operations[i])) {
            if (!copies_.try_append(storage_->range(copy->destination),
                                    storage_->range(copy->source))) {
                flush();
                if (!copies_.try_append(storage_->range(copy->destination),
                                        storage_->range(copy->source))) {
                    throw std::logic_error("Context copy cannot fit an empty batch");
                }
            }
        } else if (const auto* peer = std::get_if<PeerCopy>(&actions.operations[i])) {
            flush();
            segments_[segment].observed.work.copy_operations += enqueue_peer(*peer, execution);
        } else {
            flush();
            const auto consumed = storage_->execute(actions.operations[i], staging, execution);
            if (consumed) staging = staging.subspan(consumed);
        }
        if (segment < segments_.size() && segments_[segment].end == i + 1) {
            flush();
            timers_[segment].record_stop();
            ++segment;
        }
    }
    completed_.record(execution.stream());
    fenced_ = true;
}

std::uint32_t ContextMigration::enqueue_peer(const PeerCopy& action, CudaExecutionView execution) {
#if NINFER_ENABLE_DISTRIBUTED
    const auto member = std::find(peer_members_.begin(), peer_members_.end(), action.peer);
    const auto peer   = static_cast<std::uint32_t>(member - peer_members_.begin());
    std::size_t total = 0;
    for (const auto& range : action.ranges) total += range.bytes;
    std::size_t index = 0, offset = 0;
    std::uint32_t operations = 0;
    while (total) {
        const auto count           = std::min(total, peer_chunk_bytes_);
        const auto& first          = action.ranges[index];
        const bool direct          = first.bytes - offset >= count;
        auto payload               = direct ? storage_->range(first).subspan(offset, count)
                                            : peer_staging_.subspan(0, count);
        const auto transfer_ranges = [&](bool copy) {
            const auto flush = [&] {
                if (!copies_.size()) return;
                copies_.submit(execution);
                copies_.clear();
            };
            std::size_t used = 0;
            while (used < count) {
                const auto local = storage_->range(action.ranges[index]);
                const auto bytes = std::min(count - used, local.bytes - offset);
                if (copy) {
                    const auto piece       = local.subspan(offset, bytes);
                    const auto packed      = payload.subspan(used, bytes);
                    const auto destination = action.send ? packed : piece;
                    const auto source      = action.send ? piece : packed;
                    if (!copies_.try_append(destination, source)) {
                        flush();
                        if (!copies_.try_append(destination, source))
                            throw std::logic_error(
                                "Context peer packing cannot fit an empty copy batch");
                    }
                }
                offset += bytes;
                used += bytes;
                if (offset == local.bytes) {
                    ++index;
                    offset = 0;
                }
            }
            flush();
        };
        if (action.send) transfer_ranges(!direct);
        const distributed::PeerTransfer transfer{peer, action.send ? payload : ConstMemorySpan{},
                                                 action.send ? MemorySpan{} : payload};
        peer_group_->exchange({&transfer, 1}, execution);
        ++operations;
        if (!action.send) transfer_ranges(!direct);
        total -= count;
    }
    return operations;
#else
    (void)action;
    (void)execution;
    throw std::logic_error("Context peer transfer requires a distributed build");
#endif
}

void ContextMigration::submit(const ContextActions& actions) {
    prepare(actions);
    enqueue(actions, execution_);
}

bool ContextMigration::ready() const {
    return !occupied_ || (fenced_ && (!submitted_stream_ || completed_.ready()));
}

std::span<const runtime::ContextTransferObservation> ContextMigration::observations() {
    if (!occupied_ || !fenced_ || !ready())
        throw std::logic_error("context observations require completed execution");
    observations_.clear();
    for (std::size_t i = 0; i < segments_.size(); ++i) {
        auto observed = segments_[i].observed;
        if (submitted_stream_) {
            observed.elapsed_ns =
                std::max<std::uint64_t>(1, std::llround(timers_[i].elapsed_ms() * 1.0e6));
        }
        const auto existing =
            std::find_if(observations_.begin(), observations_.end(), [&](const auto& item) {
                return item.resource == observed.resource && item.direction == observed.direction &&
                       item.peer == observed.peer;
            });
        if (existing == observations_.end())
            observations_.push_back(observed);
        else {
            existing->work.payload_bytes += observed.work.payload_bytes;
            existing->work.copy_operations += observed.work.copy_operations;
            existing->page_groups += observed.page_groups;
            existing->image_count += observed.image_count;
            existing->elapsed_ns += observed.elapsed_ns;
        }
    }
    return observations_;
}

void ContextMigration::finish() {
    if (!occupied_ || !fenced_ || !ready())
        throw std::logic_error("cannot reuse an unfinished context action slot");
    occupied_ = false;
    fenced_   = false;
    segments_.clear();
    observations_.clear();
}

void ContextMigration::drain() {
    if (!occupied_) return;
    std::exception_ptr error;
    try {
        if (fenced_ && submitted_stream_)
            completed_.synchronize();
        else if (!fenced_ && submitted_stream_) {
            const auto status = cudaStreamSynchronize(submitted_stream_);
            if (status != cudaSuccess)
                throw std::runtime_error(std::string("context action drain failed: ") +
                                         cudaGetErrorString(status));
        }
    } catch (...) { error = std::current_exception(); }
    occupied_ = false;
    fenced_   = false;
    segments_.clear();
    observations_.clear();
    if (error) std::rethrow_exception(error);
}

} // namespace ninfer::models::context
