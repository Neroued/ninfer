#include "artifact/materializer.h"

#include "artifact/framing.h"
#include "artifact/reader.h"
#include "core/buffer.h"
#include "core/cpu/system.h"
#include "core/startup.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>

namespace ninfer::artifact {
namespace {

constexpr std::size_t kReadBlockBytes   = 64ULL * 1024 * 1024;
constexpr std::size_t kMaximumSlotCount = 4;

void populate_host_pages(MemorySpan storage) {
    if (storage.bytes < kReadBlockBytes) return;

    const CpuTopology topology;
    const auto workers = static_cast<std::uint32_t>(
        std::min<std::size_t>(16, cpu_worker_order(topology.info()).size()));
    const auto plan = plan_cpu_team(topology.info(), workers);

    struct Pages {
        std::byte* data;
        std::size_t count;
        std::uint32_t workers;
    } pages{static_cast<std::byte*>(storage.data),
            (storage.bytes + kPayloadAlignment - 1) / kPayloadAlignment, workers};

    auto threads = cpu_detail::start_workers(
        workers,
        [&](std::uint32_t rank) {
            bind_current_thread(topology, std::span(&plan.worker_cpus[rank], 1));
        },
        [](void* argument, std::uint32_t rank) noexcept {
            const auto& pages = *static_cast<Pages*>(argument);
            const auto width  = pages.count / pages.workers;
            const auto tail   = pages.count % pages.workers;
            const auto begin  = rank * width + std::min<std::size_t>(rank, tail);
            const auto end    = begin + width + (rank < tail);
            for (auto page = begin; page < end; ++page)
                pages.data[page * kPayloadAlignment] = std::byte{};
        },
        &pages);
    cpu_detail::join_workers(threads);
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw ArtifactError(std::string(operation) + ": " + cudaGetErrorName(status) + ": " +
                            cudaGetErrorString(status));
    }
}

class Slot {
public:
    explicit Slot(std::size_t bytes, bool cuda) {
        if (!cuda) {
            host = HostBuffer(bytes, kPayloadAlignment);
            return;
        }
        pinned = PinnedHostBuffer(bytes);
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

    void* data() { return event ? pinned.data() : host.data(); }

    HostBuffer host;
    PinnedHostBuffer pinned;
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
    std::size_t file             = 0;
    std::uint64_t begin          = 0;
    std::uint64_t end            = 0;
    std::byte* host              = nullptr;
    std::byte* device            = nullptr;
    std::uint64_t object_offset  = 0;
    std::uint64_t divisor_offset = 0;
    std::span<std::byte> divisors;
};

struct ReadSpan {
    std::size_t file            = 0;
    std::uint64_t begin         = 0;
    std::uint64_t end           = 0;
    std::byte* host_destination = nullptr;
};

std::vector<ReadSpan> plan_reads(std::span<const CopyRange> ranges) {
    std::vector<ReadSpan> spans;
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        const auto& range = ranges[i];
        if (i && ranges[i - 1].file == range.file && ranges[i - 1].end > range.begin) {
            throw ArtifactError("tensor source ranges overlap");
        }
        const auto begin = range.begin / kPayloadAlignment * kPayloadAlignment;
        if (spans.empty() || spans.back().file != range.file ||
            begin > align_up(spans.back().end, kPayloadAlignment, "direct range")) {
            spans.push_back({range.file, begin, range.end});
        } else {
            spans.back().end = std::max(spans.back().end, range.end);
        }
    }

    // Host-only payloads can be filled in place. Keep shared boundary pages and CPU/GPU
    // fanout in staging so every file byte is read once and GPU uploads remain asynchronous.
    std::vector<ReadSpan> reads;
    std::size_t next_range = 0;
    for (const auto& span : spans) {
        auto cursor = span.begin;
        while (next_range < ranges.size() && ranges[next_range].file == span.file &&
               ranges[next_range].begin < span.end) {
            const auto& range = ranges[next_range++];
            if (!range.host || range.device) continue;
            const auto begin = align_up(range.begin, kPayloadAlignment, "Host direct range");
            const auto end   = range.end / kPayloadAlignment * kPayloadAlignment;
            if (begin >= end) continue;
            auto* destination = range.host + begin - range.begin;
            if (reinterpret_cast<std::uintptr_t>(destination) % kPayloadAlignment) continue;
            if (cursor < begin) reads.push_back({span.file, cursor, begin});
            reads.push_back({span.file, begin, end, destination});
            cursor = end;
        }
        if (cursor < span.end) reads.push_back({span.file, cursor, span.end});
    }
    return reads;
}

void validate_divisors(std::span<float> values, const std::string& name) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto* word = reinterpret_cast<const std::byte*>(values.data() + i);
        values[i]        = std::bit_cast<float>(read_u32_le(word));
        if (!std::isfinite(values[i]) || values[i] <= 0) {
            throw ArtifactError(name + ": invalid weight divisor");
        }
    }
}

} // namespace

MaterializedArtifact& MaterializedArtifact::operator=(MaterializedArtifact&& other) noexcept {
    if (this != &other) {
        host_registration_ = {};
        objects_.clear();
        host_storage_      = std::move(other.host_storage_);
        host_registration_ = std::move(other.host_registration_);
        device_storage_    = std::move(other.device_storage_);
        objects_           = std::move(other.objects_);
        stats_             = other.stats_;
    }
    return *this;
}

CudaHostView MaterializedArtifact::host_transfer_view(ObjectHandle handle) const {
    const auto& parent    = host_parent(handle);
    const auto registered = host_registration_.view();
    if (!registered.host.bytes) { return {}; }
    const auto offset =
        static_cast<std::size_t>(parent.data - static_cast<const std::byte*>(registered.host.data));
    if (offset >= registered.host.bytes) { return {}; }
    return registered.subspan(offset, parent.geometry.bytes);
}

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
    if (handle.index >= objects_.size()) { throw ArtifactError("object has no Host backing"); }
    const auto& object = objects_[handle.index];
    if (object.host) { return {object.host->data, object.host->geometry.bytes}; }
    if (!object.host_data.empty()) { return object.host_data; }
    throw ArtifactError("object has no retained Host bytes");
}

bool MaterializedArtifact::has_device(ObjectHandle handle) const noexcept {
    return handle.index < objects_.size() && objects_[handle.index].device.has_value();
}

MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                 std::optional<CudaExecutionView> execution,
                                 const StartupObserver* startup_observer) {
    if (plan.source != &reader || plan.object_count != reader.directory().objects.size()) {
        throw ArtifactError("materialization plan belongs to another load session");
    }
    const bool cuda = !plan.device_objects.empty();
    if (plan.registered_host_bytes && !execution) {
        throw ArtifactError("registered Host weights require a CUDA execution view");
    }
    if (cuda && (!execution || execution->device() != plan.device)) {
        throw ArtifactError("device materialization requires its CUDA execution view");
    }
    std::optional<CudaDeviceScope> device_scope;
    if (cuda) { device_scope.emplace(plan.device); }
    const auto stream = cuda ? execution->stream() : nullptr;
    const StartupObserver no_observer;
    const auto& observer = startup_observer ? *startup_observer : no_observer;
    std::uint64_t total  = 0;
    for (const auto* placements :
         {&plan.device_objects, &plan.ordinary_host_objects, &plan.registered_host_objects}) {
        for (const auto& placement : *placements) {
            total = checked_add(total, placement.region.bytes, "materialized payload bytes");
        }
    }
    StartupPhaseScope phase(observer, StartupPhase::WeightsMaterialize, StartupProgressUnit::Bytes,
                            total);
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    out.stats_.file_bytes            = reader.file_bytes();
    out.stats_.read_bytes            = plan.resources.read_bytes;
    out.stats_.owned_value_bytes     = plan.resources.owned_value_bytes;
    out.stats_.device_capacity_bytes = plan.device_capacity_bytes;
    out.stats_.host_capacity_bytes   = plan.host_capacity_bytes;
    out.stats_.device_object_count   = plan.device_objects.size();
    out.stats_.host_object_count     = plan.ordinary_host_objects.size() +
                                   plan.registered_host_objects.size() +
                                   plan.resources.objects.size();
    out.host_storage_ = HostBuffer(plan.host_capacity_bytes, plan.host_alignment);
    if (cuda) { out.device_storage_ = DeviceBuffer(plan.device, plan.device_capacity_bytes); }
    populate_host_pages(out.host_storage_.span());
    if (plan.registered_host_bytes) {
        // Pin the populated destination before direct I/O, keeping page faults out of reads.
        const auto registration_start = std::chrono::steady_clock::now();
        out.host_registration_        = CudaHostRegistration(
            out.host_storage_.span().subspan(0, plan.registered_host_bytes), execution->device());
        out.stats_.registered_host_bytes = plan.registered_host_bytes;
        out.stats_.host_registration_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - registration_start)
                .count();
    }
    for (auto& resource : plan.resources.objects) {
        reader.validate_object(resource.object);
        const auto& object = reader.directory().object(resource.object);
        auto& storage      = out.objects_.at(resource.object.index);
        if (std::holds_alternative<TensorObject>(object) || !storage.host_data.empty() ||
            resource.data.size() != object_bytes(object)) {
            throw ArtifactError("invalid prepared resource");
        }
        out.stats_.retained_host_bytes += resource.data.size();
        storage.host_data = std::move(resource.data);
    }
    const auto place = [&](const std::vector<TensorPlacement>& placements, MemorySpan backing,
                           bool on_device, std::size_t alignment = 1) {
        LayoutBuilder layout;
        for (const auto& placement : placements) {
            const auto& g       = reader.geometry(placement.object);
            const auto expected = on_device
                                      ? layout.add(g.bytes, g.alignment, "weight placement")
                                      : plan_host_tensor_region(layout, reader, placement.object);
            const auto& region  = placement.region;
            if (region.offset != expected.offset || region.bytes != expected.bytes ||
                region.alignment != expected.alignment) {
                throw ArtifactError("weight placement differs from its encoded geometry");
            }
            auto& object = out.objects_.at(placement.object.index);
            auto& parent = on_device ? object.device : object.host;
            if (parent) { throw ArtifactError("duplicate tensor placement"); }
            const auto bytes = region.bind(backing);
            parent = WeightParent{g, bytes.device, static_cast<const std::byte*>(bytes.data), {}};
            if (!on_device) { out.stats_.retained_host_bytes += g.bytes; }
        }
        if (layout.finish(alignment) != backing.bytes) {
            throw ArtifactError("weight backing capacity differs from placements");
        }
    };
    place(plan.registered_host_objects,
          out.host_storage_.span().subspan(0, plan.registered_host_bytes), false,
          plan.host_alignment);
    place(plan.ordinary_host_objects, out.host_storage_.span().subspan(plan.registered_host_bytes),
          false);
    place(plan.device_objects, out.device_storage_.span(), true);
    std::vector<CopyRange> ranges;
    for (std::size_t i = 0; i < out.objects_.size(); ++i) {
        auto& object = out.objects_[i];
        if (!object.host && !object.device) { continue; }
        const auto& descriptor = reader.directory().tensor({i});
        const auto& geometry   = reader.geometry({i});
        object.weight_divisors.resize(geometry.divisor_count);
        const auto divisors = std::as_writable_bytes(std::span(object.weight_divisors));
        for (const auto& segment : reader.segments(descriptor.offset, descriptor.bytes)) {
            const auto address = [&](const std::optional<WeightParent>& parent) {
                return parent ? const_cast<std::byte*>(parent->data) + segment.destination_offset
                              : nullptr;
            };
            ranges.push_back({segment.file_index, segment.file_offset,
                              checked_add(segment.file_offset, segment.bytes, "copy range"),
                              address(object.host), address(object.device),
                              segment.destination_offset, geometry.divisor_offset, divisors});
        }
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
        return std::tie(a.file, a.begin) < std::tie(b.file, b.begin);
    });
    const auto spans = plan_reads(ranges);
    std::vector<std::unique_ptr<io::ReadOnlyFile>> files(reader.directory().files.size());
    std::uint64_t staged_bytes = 0;
    for (const auto& span : spans) {
        if (!files[span.file]) {
            files[span.file] = std::make_unique<io::ReadOnlyFile>(
                reader.open_file(span.file, io::ReadMode::Direct));
        }
        const auto bytes = align_up(span.end - span.begin, kPayloadAlignment, "direct range bytes");
        if (!span.host_destination)
            staged_bytes = checked_add(staged_bytes, bytes, "staging bytes");
    }
    const auto slot_bytes =
        static_cast<std::size_t>(std::min<std::uint64_t>(kReadBlockBytes, staged_bytes));
    std::size_t slot_count = slot_bytes ? 1 : 0;
    if (cuda && slot_bytes) {
        slot_count = static_cast<std::size_t>(
            std::min<std::uint64_t>(kMaximumSlotCount, 1 + (staged_bytes - 1) / slot_bytes));
    }
    std::vector<std::unique_ptr<Slot>> slots;
    out.stats_.peak_staging_bytes = slot_bytes * slot_count;
    std::optional<StartupPhaseScope> pin_phase;
    if (cuda) {
        pin_phase.emplace(observer, StartupPhase::WeightsStagingPin, StartupProgressUnit::Bytes,
                          out.stats_.peak_staging_bytes);
    }
    for (std::size_t i = 0; i < slot_count; ++i) {
        slots.push_back(std::make_unique<Slot>(slot_bytes, cuda));
    }
    if (pin_phase) { pin_phase->complete(); }
    TransferCompletion completion{stream, cuda};
    std::size_t next_slot = 0, next_range = 0;
    std::uint64_t copied = 0, uploaded_bytes = 0;
    const auto start = std::chrono::steady_clock::now();
    for (const auto& span : spans) {
        const auto block_bytes = span.host_destination ? kReadBlockBytes : slot_bytes;
        for (auto source = span.begin; source < span.end; source += block_bytes) {
            Slot* slot = nullptr;
            auto* destination =
                span.host_destination ? span.host_destination + source - span.begin : nullptr;
            if (!destination) {
                slot = slots[next_slot++ % slot_count].get();
                slot->wait();
                destination = static_cast<std::byte*>(slot->data());
            }
            const auto remaining = span.end - source;
            const auto request   = static_cast<std::size_t>(std::min<std::uint64_t>(
                block_bytes, align_up(remaining, kPayloadAlignment, "direct block bytes")));
            std::size_t received = 0;
            try {
                received = files[span.file]->read_direct(source, {destination, request});
            } catch (const std::system_error& e) { throw ArtifactError(e.what()); }
            if (received < std::min<std::uint64_t>(request, remaining)) {
                throw ArtifactError("direct read ended before the required payload");
            }
            out.stats_.read_bytes = checked_add(out.stats_.read_bytes, received, "read bytes");
            const auto chunk_end  = checked_add(source, received, "read block end");
            bool uploaded         = false;
            while (next_range < ranges.size() && ranges[next_range].file == span.file &&
                   ranges[next_range].begin < chunk_end) {
                const auto& range = ranges[next_range];
                const auto begin  = std::max(source, range.begin);
                const auto end    = std::min(chunk_end, range.end);
                if (begin < end) {
                    const auto* bytes = destination + begin - source;
                    if (!range.divisors.empty()) {
                        const auto object_begin = range.object_offset + begin - range.begin;
                        const auto first        = std::max(object_begin, range.divisor_offset);
                        const auto last         = std::min(object_begin + end - begin,
                                                           range.divisor_offset + range.divisors.size());
                        if (first < last) {
                            std::memcpy(range.divisors.data() + first - range.divisor_offset,
                                        bytes + first - object_begin, last - first);
                        }
                    }
                    if (range.host) {
                        auto* target = range.host + begin - range.begin;
                        if (target != bytes) std::memcpy(target, bytes, end - begin);
                        copied = checked_add(copied, end - begin, "materialized payload bytes");
                    }
                    if (range.device) {
                        check_cuda(cudaMemcpyAsync(range.device + begin - range.begin, bytes,
                                                   end - begin, cudaMemcpyHostToDevice, stream),
                                   "upload weight bytes");
                        uploaded = true;
                        copied   = checked_add(copied, end - begin, "materialized payload bytes");
                        uploaded_bytes = checked_add(uploaded_bytes, end - begin, "H2D bytes");
                    }
                }
                if (range.end > chunk_end) { break; }
                ++next_range;
            }
            if (uploaded) {
                check_cuda(cudaEventRecord(slot->event, stream),
                           "record weight staging completion");
                slot->pending = true;
            }
            phase.progress(copied, total);
        }
    }
    if (cuda) { completion.finish(); }
    if (copied != total || next_range != ranges.size()) {
        throw ArtifactError("incomplete tensor materialization");
    }
    for (std::size_t i = 0; i < out.objects_.size(); ++i) {
        auto& object = out.objects_[i];
        if (!object.host && !object.device) { continue; }
        validate_divisors(object.weight_divisors, reader.directory().tensor({i}).id);
        if (object.host) { object.host->weight_divisors = object.weight_divisors; }
        if (object.device) { object.device->weight_divisors = object.weight_divisors; }
    }
    out.stats_.h2d_bytes = uploaded_bytes;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    phase.complete(copied, total);
    return out;
}

} // namespace ninfer::artifact
