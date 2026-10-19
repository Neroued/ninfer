#include "artifact/placement.h"

#include "artifact/framing.h"
#include "artifact/reader.h"

#include <algorithm>
#include <unordered_map>

namespace ninfer::artifact {

LayoutRegion plan_host_tensor_region(LayoutBuilder& layout, const Reader& reader,
                                     ObjectHandle handle) {
    const auto& geometry = reader.geometry(handle);
    const auto& tensor   = reader.directory().tensor(handle);
    if (geometry.bytes >= 2 * kPayloadAlignment) {
        const auto source    = reader.segments(tensor.offset, 1).front().file_offset;
        const auto bias      = source % kPayloadAlignment;
        const auto alignment = std::min<std::size_t>(geometry.alignment, kPayloadAlignment);
        if (bias % alignment == 0) {
            const auto cursor = layout.finish();
            auto offset       = layout.finish(geometry.alignment);
            offset += (bias + kPayloadAlignment - offset % kPayloadAlignment) % kPayloadAlignment;
            if (offset > cursor) (void)layout.add(offset - cursor, 1, "Host I/O alignment");
        }
    }
    return layout.add(geometry.bytes, geometry.alignment, tensor.id);
}

namespace {
std::size_t tensor_hash(const EncodedTensor& t) {
    std::size_t value = static_cast<std::size_t>(t.geometry.format);
    const auto mix    = [&](std::uint64_t n) {
        value ^= n + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
    };
    mix(static_cast<std::uint64_t>(t.geometry.layout));
    for (const auto n : t.geometry.shape) mix(n);
    for (const auto& c : t.copies) {
        mix(c.object.index);
        mix(c.source);
        mix(c.destination);
        mix(c.bytes);
        mix(c.count);
        mix(c.source_stride);
        mix(c.destination_stride);
    }
    return value;
}

bool same_tensor(const EncodedTensor& a, const EncodedTensor& b) {
    return a.geometry.format == b.geometry.format && a.geometry.layout == b.geometry.layout &&
           a.geometry.shape == b.geometry.shape && a.copies == b.copies;
}
} // namespace

MaterializationPlan plan_materialization(const Reader& reader,
                                         std::span<const PlacementRequest> requests,
                                         PreparedResources resources) {
    std::vector<TensorRequest> tensors;
    tensors.reserve(requests.size());
    for (const auto& r : requests)
        tensors.push_back({whole_tensor(reader, r.object), r.device, r.host_access});
    return plan_materialization(reader, tensors, std::move(resources));
}

MaterializationPlan plan_materialization(const Reader& reader,
                                         std::span<const TensorRequest> requests,
                                         PreparedResources resources) {
    if (resources.source && resources.source != &reader)
        throw ArtifactError("prepared resources belong to another load session");
    MaterializationPlan out;
    out.source       = &reader;
    out.object_count = reader.directory().objects.size();
    out.objects.resize(out.object_count);
    out.resources = std::move(resources);
    std::unordered_multimap<std::size_t, StorageId> catalog;
    std::vector<bool> host, registered, device;
    std::vector<std::uint64_t> domains;
    bool has_device = false;
    for (const auto& request : requests) {
        const auto key =
            tensor_hash(request.tensor) ^ std::hash<std::uint64_t>{}(request.storage_domain);
        const auto [begin, end] = catalog.equal_range(key);
        auto found              = end;
        for (auto it = begin; it != end; ++it)
            if (domains[it->second.index] == request.storage_domain &&
                same_tensor(out.tensors[it->second.index], request.tensor)) {
                found = it;
                break;
            }
        StorageId id;
        if (found == end) {
            id = {out.tensors.size()};
            catalog.emplace(key, id);
            out.tensors.push_back(request.tensor);
            domains.push_back(request.storage_domain);
            host.push_back(false);
            registered.push_back(false);
            device.push_back(false);
            if (request.tensor.whole_object && !request.storage_domain)
                out.objects.at(request.tensor.whole_object->index) = id;
        } else {
            id = found->second;
        }
        out.requests.push_back(id);
        if (request.device == Device::cpu()) {
            host[id.index] = true;
            registered[id.index] =
                registered[id.index] || request.host_access == HostAccess::CudaTransfer;
        } else {
            if (request.device.type() != DeviceType::CUDA ||
                (has_device && request.device != out.device) ||
                request.host_access != HostAccess::CpuOnly)
                throw ArtifactError("materialization requires one CUDA device");
            out.device       = request.device;
            has_device       = true;
            device[id.index] = true;
        }
    }
    LayoutBuilder host_layout, registered_layout, device_layout;
    for (std::size_t i = 0; i < out.tensors.size(); ++i) {
        const auto& tensor = out.tensors[i];
        const auto& g      = tensor.geometry;
        if (host[i]) {
            auto& layout = registered[i] ? registered_layout : host_layout;
            auto& placements =
                registered[i] ? out.registered_host_objects : out.ordinary_host_objects;
            const auto region = tensor.whole_object
                                    ? plan_host_tensor_region(layout, reader, *tensor.whole_object)
                                    : layout.add(g.bytes, g.alignment, "selected Host tensor");
            placements.push_back({{i}, region});
            out.host_alignment =
                std::max({out.host_alignment, static_cast<std::size_t>(g.alignment),
                          static_cast<std::size_t>(kPayloadAlignment)});
        }
        if (device[i])
            out.device_objects.push_back(
                {{i}, device_layout.add(g.bytes, g.alignment, "device tensor")});
    }
    if (!out.registered_host_objects.empty())
        out.registered_host_bytes = registered_layout.finish(out.host_alignment);
    LayoutBuilder host_storage;
    if (out.registered_host_bytes)
        (void)host_storage.add(out.registered_host_bytes, out.host_alignment,
                               "registered Host weights");
    if (host_layout.finish())
        (void)host_storage.add(host_layout.finish(), out.host_alignment, "ordinary Host weights");
    out.host_capacity_bytes   = host_storage.finish();
    out.device_capacity_bytes = device_layout.finish();
    return out;
}

} // namespace ninfer::artifact
