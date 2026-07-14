#include "artifact/binder.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace ninfer::artifact {
namespace {

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    const std::uint64_t mask = alignment - 1;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        throw ArtifactError("materialization plan size overflows u64");
    }
    return (value + mask) & ~mask;
}

} // namespace

Binder::Binder(const Reader& reader)
    : reader_(reader), consumed_(reader.objects().size(), false),
      planned_(reader.objects().size(), false) {
    materialization_.object_count = reader.objects().size();
}

ObjectHandle Binder::find_unconsumed(std::string_view name) {
    const auto& objects = reader_.objects();
    const ObjectDescriptor* object = reader_.find(name);
    if (object == nullptr) {
        throw ArtifactError("required artifact object is missing: " + std::string(name));
    }
    const auto index = static_cast<std::size_t>(object - objects.data());
    if (consumed_[index]) {
        throw ArtifactError("artifact object was bound more than once: " + std::string(name));
    }
    consumed_[index] = true;
    return ObjectHandle{index};
}

ObjectHandle Binder::require_tensor(std::string_view name, NumericFormat format,
                                    StorageLayout layout,
                                    std::span<const std::uint64_t> shape) {
    const ObjectHandle handle = find_unconsumed(name);
    const auto* tensor = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("required tensor is a resource: " + std::string(name));
    }
    if (tensor->format != format || tensor->layout != layout ||
        !std::equal(tensor->shape.begin(), tensor->shape.end(), shape.begin(), shape.end())) {
        throw ArtifactError("tensor descriptor does not match target contract: " +
                            std::string(name));
    }
    return handle;
}

ObjectHandle Binder::require_resource(std::string_view name, ResourceEncoding encoding) {
    const ObjectHandle handle = find_unconsumed(name);
    const auto* resource = std::get_if<ResourceDescriptor>(&descriptor(handle));
    if (resource == nullptr) {
        throw ArtifactError("required resource is a tensor: " + std::string(name));
    }
    if (resource->encoding != encoding) {
        throw ArtifactError("resource encoding does not match target contract: " +
                            std::string(name));
    }
    return handle;
}

const ObjectDescriptor& Binder::descriptor(ObjectHandle handle) const {
    if (handle.index >= reader_.objects().size()) {
        throw ArtifactError("artifact object handle is out of range");
    }
    return reader_.objects()[handle.index];
}

PayloadSpan Binder::payload(ObjectHandle handle) const {
    return reader_.payload(descriptor(handle));
}

void Binder::materialize_on_device(ObjectHandle handle) {
    const auto* tensor = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("resource cannot be materialized as a device tensor");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(tensor->name));
    }
    const std::uint64_t alignment = tensor_alignment(tensor->layout);
    const std::uint64_t offset =
        align_up(materialization_.device_capacity_bytes, alignment);
    if (tensor->bytes > std::numeric_limits<std::uint64_t>::max() - offset) {
        throw ArtifactError("materialization plan size overflows u64");
    }
    materialization_.device_objects.push_back(
        DeviceMaterialization{handle, offset, tensor->bytes, alignment});
    materialization_.device_capacity_bytes = offset + tensor->bytes;
    planned_[handle.index] = true;
}

void Binder::retain_on_host(ObjectHandle handle) {
    const auto* resource = std::get_if<ResourceDescriptor>(&descriptor(handle));
    if (resource == nullptr) {
        throw ArtifactError("tensor cannot be retained as a host resource");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(resource->name));
    }
    materialization_.host_objects.push_back(HostMaterialization{handle});
    planned_[handle.index] = true;
}

MaterializationPlan Binder::finish() {
    const auto it = std::find(consumed_.begin(), consumed_.end(), false);
    if (it != consumed_.end()) {
        const auto index = static_cast<std::size_t>(it - consumed_.begin());
        throw ArtifactError("artifact object was not consumed by the selected target: " +
                            std::string(object_name(reader_.objects()[index])));
    }
    const auto unplanned = std::find(planned_.begin(), planned_.end(), false);
    if (unplanned != planned_.end()) {
        const auto index = static_cast<std::size_t>(unplanned - planned_.begin());
        throw ArtifactError("artifact object has no materialization placement: " +
                            std::string(object_name(reader_.objects()[index])));
    }
    return std::move(materialization_);
}

} // namespace ninfer::artifact
