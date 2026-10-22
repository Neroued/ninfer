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
    static std::size_t allocation_bytes(std::size_t bytes, std::size_t alignment, bool cuda) {
        return cuda ? checked_add(bytes, alignment - 1, "aligned weight staging") : bytes;
    }

    Slot(std::size_t bytes, std::size_t alignment, bool cuda) {
        if (!cuda) {
            host_ = HostBuffer(bytes, alignment);
            data_ = static_cast<std::byte*>(host_.data());
            return;
        }
        pinned_            = PinnedHostBuffer(allocation_bytes(bytes, alignment, cuda));
        const auto address = reinterpret_cast<std::uintptr_t>(pinned_.data());
        const auto padding = (alignment - address % alignment) % alignment;
        data_              = static_cast<std::byte*>(pinned_.data()) + padding;
        check_cuda(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming),
                   "create weight staging completion event");
    }

    ~Slot() {
        if (pending_) { (void)cudaEventSynchronize(event_); }
        if (event_) { (void)cudaEventDestroy(event_); }
    }

    void wait() {
        if (pending_) {
            check_cuda(cudaEventSynchronize(event_), "wait for weight staging transfer");
            pending_ = false;
        }
    }

    void record(cudaStream_t stream) {
        check_cuda(cudaEventRecord(event_, stream), "record weight staging completion");
        pending_ = true;
    }

    std::byte* data() const noexcept { return data_; }

private:
    HostBuffer host_;
    PinnedHostBuffer pinned_;
    std::byte* data_   = nullptr;
    cudaEvent_t event_ = nullptr;
    bool pending_      = false;
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
    std::uint64_t pattern_offset = 0, row_bytes = 0, rows = 1;
    std::uint64_t source_stride = 0, destination_stride = 0;
};

struct ReadSpan {
    std::size_t file = 0;
    io::DirectReadWindow window{};
    std::byte* host_destination = nullptr;

    std::size_t block_bytes(io::FileView source) const {
        const auto alignment = host_destination
                                   ? std::max(source.offset_alignment, source.memory_alignment)
                                   : source.offset_alignment;
        return std::min(window.bytes, static_cast<std::size_t>(align_up(kReadBlockBytes, alignment,
                                                                        "direct read block")));
    }
};

std::vector<ReadSpan> plan_reads(std::span<const CopyRange> ranges,
                                 std::span<const io::FileView> files) {
    std::vector<ReadSpan> spans;
    bool overlaps          = false;
    std::uint64_t furthest = 0;
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        const auto& range = ranges[i];
        if (!i || ranges[i - 1].file != range.file) furthest = 0;
        overlaps |= range.begin < furthest;
        furthest = std::max(furthest, range.end);
        const auto window =
            io::direct_read_window(files[range.file], range.begin, range.end - range.begin);
        if (spans.empty() || spans.back().file != range.file ||
            window.offset > spans.back().window.offset + spans.back().window.bytes) {
            spans.push_back({range.file, window});
        } else {
            auto& previous   = spans.back().window;
            const auto begin = previous.offset + previous.prefix;
            const auto end   = std::max(previous.offset + previous.required, range.end);
            previous         = io::direct_read_window(files[range.file], begin, end - begin);
        }
    }

    // Host-only payloads can be filled in place. Keep shared boundary pages and CPU/GPU
    // fanout in staging so every file byte is read once and GPU uploads remain asynchronous.
    if (overlaps) return spans;
    std::vector<ReadSpan> reads;
    std::size_t next_range = 0;
    for (const auto& span : spans) {
        const auto file         = files[span.file];
        const auto required_end = span.window.offset + span.window.required;
        auto cursor             = span.window.offset;
        while (next_range < ranges.size() && ranges[next_range].file == span.file &&
               ranges[next_range].begin < required_end) {
            const auto& range = ranges[next_range++];
            if (!range.host || range.device || range.rows != 1) continue;
            auto begin     = align_up(range.begin, file.offset_alignment, "Host direct range");
            const auto end = range.end / file.offset_alignment * file.offset_alignment;
            if (begin >= end) continue;
            auto* destination  = range.host + range.pattern_offset + begin - range.begin;
            const auto address = reinterpret_cast<std::uintptr_t>(destination);
            const auto padding =
                (file.memory_alignment - address % file.memory_alignment) % file.memory_alignment;
            if (padding % file.offset_alignment || padding >= end - begin) continue;
            begin += padding;
            destination += padding;
            if (cursor < begin)
                reads.push_back({span.file, io::direct_read_window(file, cursor, begin - cursor)});
            reads.push_back(
                {span.file, io::direct_read_window(file, begin, end - begin), destination});
            cursor = end;
        }
        if (cursor < required_end)
            reads.push_back(
                {span.file, io::direct_read_window(file, cursor, required_end - cursor)});
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
        resources_.clear();
        tensors_.clear();
        host_storage_      = std::move(other.host_storage_);
        host_registration_ = std::move(other.host_registration_);
        device_storage_    = std::move(other.device_storage_);
        resources_         = std::move(other.resources_);
        tensors_           = std::move(other.tensors_);
        object_storage_    = std::move(other.object_storage_);
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
    return *tensors_[object_storage_[handle.index]->index].device;
}

const WeightParent& MaterializedArtifact::parent(StorageId id, Device device) const {
    const auto& tensor = tensors_.at(id.index);
    const auto& value  = device == Device::cpu() ? tensor.host : tensor.device;
    if (!value || value->device != device)
        throw ArtifactError("tensor has no backing on requested device");
    return *value;
}

const WeightParent& MaterializedArtifact::host_parent(ObjectHandle handle) const {
    if (handle.index >= object_storage_.size() || !object_storage_[handle.index])
        throw ArtifactError("object has no Host weight backing");
    return parent(*object_storage_[handle.index], Device::cpu());
}

std::span<const std::byte> MaterializedArtifact::host_bytes(ObjectHandle handle) const {
    if (handle.index >= resources_.size()) { throw ArtifactError("object has no Host backing"); }
    const auto& bytes = resources_[handle.index];
    if (object_storage_[handle.index]) {
        const auto& tensor = tensors_[object_storage_[handle.index]->index];
        if (tensor.host) return {tensor.host->data, tensor.host->geometry.bytes};
    }
    if (!bytes.empty()) { return bytes; }
    throw ArtifactError("object has no retained Host bytes");
}

bool MaterializedArtifact::has_device(ObjectHandle handle) const noexcept {
    return handle.index < object_storage_.size() && object_storage_[handle.index] &&
           tensors_[object_storage_[handle.index]->index].device.has_value();
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
            for (const auto& copy : plan.tensors.at(placement.storage.index).copies)
                total = checked_add(total, checked_mul(copy.bytes, copy.count, "copy payload"),
                                    "materialized payload bytes");
        }
    }
    StartupPhaseScope phase(observer, StartupPhase::WeightsMaterialize, StartupProgressUnit::Bytes,
                            total);
    MaterializedArtifact out;
    out.resources_.resize(plan.object_count);
    out.tensors_.resize(plan.tensors.size());
    out.object_storage_              = plan.objects;
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
        auto& storage      = out.resources_.at(resource.object.index);
        if (std::holds_alternative<TensorObject>(object) || !storage.empty() ||
            resource.data.size() != object_bytes(object)) {
            throw ArtifactError("invalid prepared resource");
        }
        out.stats_.retained_host_bytes += resource.data.size();
        storage = std::move(resource.data);
    }
    const auto place = [&](const std::vector<TensorPlacement>& placements, MemorySpan backing,
                           bool on_device, std::size_t alignment = 1) {
        LayoutBuilder layout;
        for (const auto& placement : placements) {
            const auto& tensor = plan.tensors.at(placement.storage.index);
            const auto& g      = tensor.geometry;
            const auto expected =
                !on_device && tensor.whole_object
                    ? plan_host_tensor_region(layout, reader, *tensor.whole_object)
                    : layout.add(g.bytes, g.alignment, "weight placement");
            const auto& region = placement.region;
            if (region.offset != expected.offset || region.bytes != expected.bytes ||
                region.alignment != expected.alignment) {
                throw ArtifactError("weight placement differs from its encoded geometry");
            }
            auto& object = out.tensors_.at(placement.storage.index);
            auto& parent = on_device ? object.device : object.host;
            if (parent) { throw ArtifactError("duplicate tensor placement"); }
            const auto bytes = region.bind(backing);
            parent = WeightParent{g, bytes.device, static_cast<const std::byte*>(bytes.data), {}};
            if (!on_device) { out.stats_.retained_host_bytes += g.bytes; }
            if (!tensor.whole_object) {
                if (on_device)
                    check_cuda(cudaMemsetAsync(bytes.data, 0, bytes.bytes, stream),
                               "initialize encoded padding");
                else
                    std::memset(bytes.data, 0, bytes.bytes);
            }
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
    for (std::size_t i = 0; i < out.tensors_.size(); ++i) {
        auto& object       = out.tensors_[i];
        const auto& tensor = plan.tensors[i];
        object.weight_divisors.resize(tensor.geometry.divisor_count);
        const auto divisors = std::as_writable_bytes(std::span(object.weight_divisors));
        for (const auto& copy : tensor.copies) {
            const auto& source = reader.directory().tensor(copy.object);
            if (!copy.count || !copy.bytes) throw ArtifactError("empty encoded copy");
            const auto extent =
                checked_add(checked_mul(copy.count - 1, copy.source_stride, "encoded copy extent"),
                            copy.bytes, "encoded copy extent");
            if (copy.source > source.bytes || extent > source.bytes - copy.source ||
                copy.destination > tensor.geometry.bytes ||
                checked_add(checked_mul(copy.count - 1, copy.destination_stride, "copy extent"),
                            copy.bytes, "copy extent") > tensor.geometry.bytes - copy.destination)
                throw ArtifactError("encoded copy exceeds backing");
            for (const auto& segment : reader.segments(source.offset + copy.source, extent)) {
                const auto address = [&](const std::optional<WeightParent>& parent) {
                    return parent ? const_cast<std::byte*>(parent->data) + copy.destination
                                  : nullptr;
                };
                ranges.push_back(
                    {segment.file_index, segment.file_offset, segment.file_offset + segment.bytes,
                     address(object.host), address(object.device), copy.destination,
                     tensor.geometry.divisor_offset, divisors, segment.destination_offset,
                     copy.bytes, copy.count, copy.source_stride, copy.destination_stride});
            }
        }
    }
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
        return std::tie(a.file, a.begin) < std::tie(b.file, b.begin);
    });
    std::vector<std::unique_ptr<io::ReadOnlyFile>> files(reader.directory().files.size());
    std::vector<io::FileView> file_views(files.size());
    for (const auto& range : ranges) {
        if (files[range.file]) continue;
        files[range.file] =
            std::make_unique<io::ReadOnlyFile>(reader.open_file(range.file, io::ReadMode::Direct));
        file_views[range.file] = files[range.file]->view();
    }
    const auto spans           = plan_reads(ranges, file_views);
    std::uint64_t staged_bytes = 0;
    std::size_t slot_bytes = 0, slot_alignment = 1;
    for (const auto& span : spans) {
        if (span.host_destination) continue;
        const auto file = file_views[span.file];
        staged_bytes    = checked_add(staged_bytes, span.window.bytes, "staging bytes");
        slot_bytes      = std::max(slot_bytes, span.block_bytes(file));
        slot_alignment  = std::max(slot_alignment, file.memory_alignment);
    }
    std::size_t slot_count = slot_bytes ? 1 : 0;
    if (cuda && slot_bytes) {
        slot_count = static_cast<std::size_t>(
            std::min<std::uint64_t>(kMaximumSlotCount, 1 + (staged_bytes - 1) / slot_bytes));
    }
    std::vector<std::unique_ptr<Slot>> slots;
    out.stats_.peak_staging_bytes = checked_mul(
        Slot::allocation_bytes(slot_bytes, slot_alignment, cuda), slot_count, "staging backing");
    std::optional<StartupPhaseScope> pin_phase;
    if (cuda) {
        pin_phase.emplace(observer, StartupPhase::WeightsStagingPin, StartupProgressUnit::Bytes,
                          out.stats_.peak_staging_bytes);
    }
    for (std::size_t i = 0; i < slot_count; ++i) {
        slots.push_back(std::make_unique<Slot>(slot_bytes, slot_alignment, cuda));
    }
    if (pin_phase) { pin_phase->complete(); }
    TransferCompletion completion{stream, cuda};
    std::size_t next_slot = 0, next_range = 0;
    std::uint64_t copied = 0, uploaded_bytes = 0;
    const auto start = std::chrono::steady_clock::now();
    for (const auto& span : spans) {
        const auto block_bytes = span.block_bytes(file_views[span.file]);
        for (std::size_t offset = 0; offset < span.window.bytes; offset += block_bytes) {
            const auto source = span.window.offset + offset;
            Slot* slot        = nullptr;
            auto* destination = span.host_destination ? span.host_destination + offset : nullptr;
            if (!destination) {
                slot = slots[next_slot++ % slot_count].get();
                slot->wait();
                destination = slot->data();
            }
            const auto request   = std::min(block_bytes, span.window.bytes - offset);
            std::size_t received = 0;
            try {
                received = files[span.file]->read_direct(source, {destination, request});
            } catch (const std::system_error& e) { throw ArtifactError(e.what()); }
            const auto chunk_end = checked_add(source, received, "read block end");
            if (received < request && chunk_end < span.window.offset + span.window.required) {
                throw ArtifactError("direct read ended before the required payload");
            }
            out.stats_.read_bytes = checked_add(out.stats_.read_bytes, received, "read bytes");
            bool uploaded         = false;
            while (next_range < ranges.size() &&
                   (ranges[next_range].file < span.file ||
                    (ranges[next_range].file == span.file && ranges[next_range].end <= source)))
                ++next_range;
            for (auto index = next_range;
                 index < ranges.size() && ranges[index].file == span.file &&
                 ranges[index].begin < chunk_end;
                 ++index) {
                const auto& range = ranges[index];
                const auto begin  = std::max(source, range.begin),
                           end    = std::min(chunk_end, range.end);
                if (begin >= end) continue;
                const auto first_byte = range.pattern_offset + begin - range.begin;
                const auto last_byte  = range.pattern_offset + end - range.begin;
                auto row              = range.rows == 1 ? 0 : first_byte / range.source_stride;
                while (row < range.rows) {
                    const auto row_begin = row * range.source_stride;
                    if (row_begin >= last_byte) break;
                    const auto first = std::max(first_byte, row_begin);
                    const auto last  = std::min(last_byte, row_begin + range.row_bytes);
                    if (first >= last) {
                        ++row;
                        continue;
                    }
                    auto count = std::uint64_t{1};
                    if (range.rows > 1 && first == row_begin && last == row_begin + range.row_bytes)
                        count = std::min(range.rows - row,
                                         1 + (last_byte - row_begin - range.row_bytes) /
                                                 range.source_stride);
                    const auto offset = row * range.destination_stride + first - row_begin;
                    const auto* bytes = destination + begin - source + first - first_byte;
                    const auto width  = last - first;
                    if (!range.divisors.empty()) {
                        for (std::uint64_t n = 0; n < count; ++n) {
                            const auto object_begin =
                                range.object_offset + offset + n * range.destination_stride;
                            const auto a = std::max(object_begin, range.divisor_offset);
                            const auto b = std::min(object_begin + width,
                                                    range.divisor_offset + range.divisors.size());
                            if (a < b)
                                std::memcpy(range.divisors.data() + a - range.divisor_offset,
                                            bytes + n * range.source_stride + a - object_begin,
                                            b - a);
                        }
                    }
                    if (range.host) {
                        for (std::uint64_t n = 0; n < count; ++n) {
                            auto* target      = range.host + offset + n * range.destination_stride;
                            const auto* input = bytes + n * range.source_stride;
                            if (target != input) std::memcpy(target, input, width);
                        }
                        copied += width * count;
                    }
                    if (range.device) {
                        if (count == 1)
                            check_cuda(cudaMemcpyAsync(range.device + offset, bytes, width,
                                                       cudaMemcpyHostToDevice, stream),
                                       "upload weight bytes");
                        else
                            check_cuda(cudaMemcpy2DAsync(range.device + offset,
                                                         range.destination_stride, bytes,
                                                         range.source_stride, width, count,
                                                         cudaMemcpyHostToDevice, stream),
                                       "upload weight rows");
                        uploaded = true;
                        copied += width * count;
                        uploaded_bytes += width * count;
                    }
                    row += count;
                }
            }
            if (uploaded) { slot->record(stream); }
            phase.progress(copied, total);
        }
    }
    if (cuda) { completion.finish(); }
    if (copied != total) { throw ArtifactError("incomplete tensor materialization"); }
    for (std::size_t i = 0; i < out.tensors_.size(); ++i) {
        auto& object = out.tensors_[i];
        if (!object.host && !object.device) { continue; }
        validate_divisors(object.weight_divisors, "materialized tensor " + std::to_string(i));
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
