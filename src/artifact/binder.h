#pragma once

#include "artifact/reader.h"

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

struct ObjectHandle {
    std::size_t index = 0;
};

struct DeviceMaterialization {
    ObjectHandle object;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
    std::uint64_t alignment = 0;
};

struct HostMaterialization {
    ObjectHandle object;
};

struct MaterializationPlan {
    std::size_t object_count = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::vector<DeviceMaterialization> device_objects;
    std::vector<HostMaterialization> host_objects;
};

class Binder {
public:
    explicit Binder(const Reader& reader);

    ObjectHandle require_tensor(std::string_view name, NumericFormat format,
                                StorageLayout layout,
                                std::span<const std::uint64_t> shape);
    ObjectHandle require_resource(std::string_view name, ResourceEncoding encoding);

    const ObjectDescriptor& descriptor(ObjectHandle handle) const;
    PayloadSpan payload(ObjectHandle handle) const;
    void materialize_on_device(ObjectHandle handle);
    void retain_on_host(ObjectHandle handle);
    MaterializationPlan finish();

private:
    ObjectHandle find_unconsumed(std::string_view name);

    const Reader& reader_;
    std::vector<bool> consumed_;
    std::vector<bool> planned_;
    MaterializationPlan materialization_;
};

} // namespace ninfer::artifact
