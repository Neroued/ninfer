#include "artifact/materializer.h"

#include "artifact/framing.h"
#include "artifact/reader.h"
#include "core/startup.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>

namespace ninfer::artifact {
namespace {

constexpr std::size_t kSlotBytes        = 64ULL * 1024 * 1024;
constexpr std::size_t kMaximumSlotCount = 4;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw ArtifactError(std::string(operation) + ": " + cudaGetErrorName(status) + ": " +
                            cudaGetErrorString(status));
    }
}

class Slot {
public:
    explicit Slot(std::size_t bytes) : buffer(bytes) {
        check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                   "create weight staging completion event");
    }

    ~Slot() {
        if (pending) { (void)cudaEventSynchronize(event); }
        if (event) { (void)cudaEventDestroy(event); }
    }

    void wait() {
        if (pending) {
            check_cuda(cudaEventSynchronize(event), "wait for weight staging transfer");
            pending = false;
        }
    }

    PinnedHostBuffer buffer;
    cudaEvent_t event = nullptr;
    bool pending      = false;
};

// Also covers failure between a queued copy and its event record. Destruct before slots.
struct TransferCompletion {
    cudaStream_t stream;
    bool pending = true;

    ~TransferCompletion() {
        if (pending) { (void)cudaStreamSynchronize(stream); }
    }

    void finish() {
        check_cuda(cudaStreamSynchronize(stream), "complete weight upload");
        pending = false;
    }
};

struct CopyRange {
    std::size_t file       = 0;
    std::uint64_t begin    = 0;
    std::uint64_t end      = 0;
    std::byte* destination = nullptr;
};

struct ReadSpan {
    std::size_t file    = 0;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

float read_divisor(const Reader& reader, ObjectHandle handle, const WeightGeometry& geometry,
                   std::span<const std::byte> host, MaterializationStats& stats) {
    if (geometry.format != QType::NVFP4) { return 0.0F; }
    std::array<std::byte, 4> word{};
    if (!host.empty()) {
        std::copy_n(host.data() + geometry.divisor_offset, word.size(), word.data());
    } else {
        const auto& object = reader.directory().tensor(handle);
        reader.read_into(checked_add(object.offset, geometry.divisor_offset, "weight divisor"),
                         word);
        stats.read_bytes = checked_add(stats.read_bytes, word.size(), "read bytes");
    }
    const auto value = std::bit_cast<float>(read_u32_le(word.data()));
    if (!std::isfinite(value) || value <= 0) {
        throw ArtifactError(reader.directory().tensor(handle).id +
                            ": invalid NVFP4 weight divisor");
    }
    return value;
}

} // namespace

const WeightParent& MaterializedArtifact::device_parent(ObjectHandle handle) const {
    if (!has_device(handle)) { throw ArtifactError("object has no device weight backing"); }
    return *objects_[handle.index].device;
}

const WeightParent& MaterializedArtifact::host_parent(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || !objects_[handle.index].host) {
        throw ArtifactError("object has no Host weight backing");
    }
    return *objects_[handle.index].host;
}

std::span<const std::byte> MaterializedArtifact::host_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].host_data.empty()) {
        throw ArtifactError("object has no retained Host bytes");
    }
    return objects_[handle.index].host_data;
}

bool MaterializedArtifact::has_device(ObjectHandle handle) const noexcept {
    return handle.index < objects_.size() && objects_[handle.index].device.has_value();
}

MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                 DeviceContext& device, const StartupObserver* startup_observer) {
    if (plan.source != &reader || plan.object_count != reader.directory().objects.size()) {
        throw ArtifactError("materialization plan belongs to another load session");
    }
    const StartupObserver no_observer;
    const auto& observer = startup_observer ? *startup_observer : no_observer;
    std::uint64_t total  = 0;
    for (const auto& placement : plan.device_objects) {
        total = checked_add(total, placement.bytes, "device payload bytes");
    }
    StartupPhaseScope phase(observer, StartupPhase::WeightsMaterialize, StartupProgressUnit::Bytes,
                            total);
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    out.stats_.file_bytes            = reader.file_bytes();
    out.stats_.read_bytes            = plan.prior_read_bytes;
    out.stats_.owned_value_bytes     = plan.owned_value_bytes;
    out.stats_.device_capacity_bytes = plan.device_capacity_bytes;
    out.stats_.device_object_count   = plan.device_objects.size();
    out.stats_.host_object_count     = plan.host_objects.size();
    if (plan.device_capacity_bytes > std::numeric_limits<std::size_t>::max()) {
        throw ArtifactError("device backing exceeds size_t");
    }
    if (plan.device_capacity_bytes) {
        out.arena_ =
            std::make_unique<DeviceArena>(static_cast<std::size_t>(plan.device_capacity_bytes));
    }
    for (auto& placement : plan.host_objects) {
        reader.validate_object(placement.object);
        auto& storage = out.objects_.at(placement.object.index);
        if (!storage.host_data.empty()) { throw ArtifactError("duplicate Host placement"); }
        const auto& object = reader.directory().object(placement.object);
        if (placement.data.empty()) {
            placement.data = reader.read_object(placement.object);
            out.stats_.read_bytes =
                checked_add(out.stats_.read_bytes, placement.data.size(), "Host read bytes");
        }
        if (placement.data.size() != object_bytes(object)) {
            throw ArtifactError("Host placement size differs from object");
        }
        storage.host_data = std::move(placement.data);
        out.stats_.retained_host_bytes =
            checked_add(out.stats_.retained_host_bytes, storage.host_data.size(), "retained bytes");
        if (std::holds_alternative<TensorObject>(object)) {
            const auto& geometry = reader.geometry(placement.object);
            const auto divisor =
                read_divisor(reader, placement.object, geometry, storage.host_data, out.stats_);
            storage.host = WeightParent{geometry, storage.host_data.data(), divisor};
        }
    }
    std::vector<CopyRange> ranges;
    for (const auto& placement : plan.device_objects) {
        const auto& geometry = reader.geometry(placement.object);
        auto& object         = out.objects_.at(placement.object.index);
        if (object.device || !out.arena_ || geometry.bytes != placement.bytes) {
            throw ArtifactError("invalid or duplicate device placement");
        }
        auto storage      = out.arena_->alloc_bytes(static_cast<std::size_t>(placement.bytes),
                                                    static_cast<std::size_t>(placement.alignment));
        const auto offset = static_cast<std::uint64_t>(static_cast<std::byte*>(storage.data) -
                                                       static_cast<std::byte*>(out.arena_->base()));
        if (offset != placement.offset) {
            throw ArtifactError("device offset differs from materialization plan");
        }
        const auto divisor = object.host
                                 ? object.host->weight_scale_divisor
                                 : read_divisor(reader, placement.object, geometry, {}, out.stats_);
        object.device =
            WeightParent{geometry, static_cast<const std::byte*>(storage.data), divisor};
        const auto& descriptor = reader.directory().tensor(placement.object);
        for (const auto& segment : reader.segments(descriptor.offset, descriptor.bytes)) {
            ranges.push_back({segment.file_index, segment.file_offset,
                              checked_add(segment.file_offset, segment.bytes, "copy range"),
                              static_cast<std::byte*>(storage.data) + segment.destination_offset});
        }
    }
    if (ranges.empty()) {
        phase.complete();
        return out;
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
        return std::tie(a.file, a.begin) < std::tie(b.file, b.begin);
    });
    std::vector<ReadSpan> spans;
    std::uint64_t aligned_bytes = 0;
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        const auto& range = ranges[i];
        if (i && ranges[i - 1].file == range.file && ranges[i - 1].end > range.begin) {
            throw ArtifactError("device source ranges overlap");
        }
        const auto begin = range.begin / kPayloadAlignment * kPayloadAlignment;
        if (spans.empty() || spans.back().file != range.file ||
            begin > align_up(spans.back().end, kPayloadAlignment, "direct range")) {
            spans.push_back({range.file, begin, range.end});
        } else {
            spans.back().end = std::max(spans.back().end, range.end);
        }
    }
    for (const auto& span : spans) {
        aligned_bytes = checked_add(
            aligned_bytes, align_up(span.end - span.begin, kPayloadAlignment, "direct range bytes"),
            "direct bytes");
    }
    const auto slot_bytes =
        static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_bytes));
    const auto slot_count = static_cast<std::size_t>(
        std::min<std::uint64_t>(kMaximumSlotCount, 1 + (aligned_bytes - 1) / slot_bytes));
    std::vector<std::unique_ptr<Slot>> slots;
    out.stats_.peak_staging_bytes = slot_bytes * slot_count;
    StartupPhaseScope pin_phase(observer, StartupPhase::WeightsStagingPin,
                                StartupProgressUnit::Bytes, out.stats_.peak_staging_bytes);
    for (std::size_t i = 0; i < slot_count; ++i) {
        slots.push_back(std::make_unique<Slot>(slot_bytes));
    }
    pin_phase.complete();
    TransferCompletion completion{device.transfer_stream};
    std::size_t next_slot  = 0;
    std::size_t next_range = 0;
    std::uint64_t copied   = 0;
    const auto start       = std::chrono::steady_clock::now();
    for (const auto& span : spans) {
        for (auto source = span.begin; source < span.end; source += slot_bytes) {
            auto& slot = *slots[next_slot++ % slot_count];
            slot.wait();
            const auto remaining = span.end - source;
            const auto request   = static_cast<std::size_t>(std::min<std::uint64_t>(
                slot_bytes, align_up(remaining, kPayloadAlignment, "direct block bytes")));
            const auto received  = reader.read_direct(
                span.file, source, {static_cast<std::byte*>(slot.buffer.data()), request});
            if (received < std::min<std::uint64_t>(request, remaining)) {
                throw ArtifactError("direct read ended before the required payload");
            }
            out.stats_.read_bytes = checked_add(out.stats_.read_bytes, received, "read bytes");
            const auto chunk_end  = checked_add(source, received, "read block end");
            while (next_range < ranges.size() && ranges[next_range].file == span.file &&
                   ranges[next_range].begin < chunk_end) {
                const auto& range = ranges[next_range];
                const auto begin  = std::max(source, range.begin);
                const auto end    = std::min(chunk_end, range.end);
                if (begin < end) {
                    check_cuda(cudaMemcpyAsync(range.destination + (begin - range.begin),
                                               static_cast<const std::byte*>(slot.buffer.data()) +
                                                   (begin - source),
                                               static_cast<std::size_t>(end - begin),
                                               cudaMemcpyHostToDevice, device.transfer_stream),
                               "upload weight bytes");
                    copied = checked_add(copied, end - begin, "copied bytes");
                }
                if (range.end > chunk_end) { break; }
                ++next_range;
            }
            check_cuda(cudaEventRecord(slot.event, device.transfer_stream),
                       "record weight staging completion");
            slot.pending = true;
            phase.progress(copied, total);
        }
    }
    for (const auto& slot : slots) { slot->wait(); }
    completion.finish();
    if (copied != total || next_range != ranges.size()) {
        throw ArtifactError("incomplete device upload");
    }
    out.stats_.h2d_bytes = copied;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    slots.clear();
    phase.complete(copied, total);
    return out;
}

} // namespace ninfer::artifact
