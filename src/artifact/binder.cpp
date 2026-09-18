#include "artifact/binder.h"

#include "artifact/framing.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace ninfer::artifact {

float HostValues::scalar_f32() const {
    if (format != QType::FP32 || elements != 1 || data.size() != 4) {
        throw ArtifactError("auxiliary requires one represented FP32 value");
    }
    return std::bit_cast<float>(read_u32_le(data.data()));
}

std::vector<std::int32_t> HostValues::integers() const {
    if (format != QType::INT32 || data.size() != checked_mul(elements, 4, "integer values")) {
        throw ArtifactError("semantic table requires INT32 values");
    }
    std::vector<std::int32_t> out;
    out.reserve(static_cast<std::size_t>(elements));
    for (std::size_t i = 0; i < elements; ++i) {
        out.push_back(std::bit_cast<std::int32_t>(read_u32_le(data.data() + i * 4)));
    }
    return out;
}

Binder::Binder(const Reader& reader)
    : reader_(reader), demands_(reader.directory().objects.size()) {}

ParameterReference Binder::parameter(std::string_view name, Shape shape, Residency residency,
                                     std::optional<QType> exact_format) {
    const auto found = reader_.directory().bindings.find(name);
    if (found == reader_.directory().bindings.end()) {
        throw ArtifactError("missing logical parameter " + std::string(name));
    }
    return binding(std::string(name), found->second, std::move(shape), residency, exact_format);
}

ParameterReference Binder::binding(std::string name, const Binding& binding, Shape shape,
                                   Residency residency, std::optional<QType> exact_format) {
    if (weight_element_count(shape) != binding.elements ||
        (binding.whole_object &&
         reader_.directory().tensor(binding.parts.at(0).object).shape != shape)) {
        throw ArtifactError(name + ": logical shape differs from Binding coverage");
    }
    for (const auto& part : binding.parts) {
        const auto& geometry = reader_.geometry(part.object);
        if (exact_format && geometry.format != *exact_format) {
            throw ArtifactError(name +
                                ": representation does not match its mathematical value type");
        }
        if (residency == Residency::Device) {
            require_device(part.object);
        } else if (residency == Residency::Host) {
            (void)host_object(part.object);
        }
    }
    return {std::move(name), std::move(shape), binding, residency};
}

const Use& Binder::use(std::string_view parameter, std::string_view input) const {
    const auto found = reader_.directory().uses.find({std::string(parameter), std::string(input)});
    if (found == reader_.directory().uses.end()) {
        throw ArtifactError("missing Use " + std::string(parameter) + "@" + std::string(input));
    }
    return found->second;
}

bool Binder::contains(std::string_view parameter) const {
    return reader_.directory().bindings.contains(parameter);
}

void Binder::require_device(ObjectHandle object, std::uint64_t alignment) {
    const auto& geometry = reader_.geometry(object);
    if (!alignment || (alignment & (alignment - 1))) {
        throw ArtifactError("device alignment must be a power of two");
    }
    auto& demand     = demands_.at(object.index);
    demand.device    = true;
    demand.alignment = std::max({demand.alignment, alignment, geometry.alignment});
}

std::span<const std::byte> Binder::host_object(ObjectHandle object) {
    reader_.validate_object(object);
    auto& demand = demands_.at(object.index);
    if (!demand.host) {
        demand.host_data = reader_.read_object(object);
        read_bytes_      = checked_add(read_bytes_, demand.host_data.size(), "Host read bytes");
        demand.host      = true;
    }
    return demand.host_data;
}

ObjectHandle Binder::resource(std::string_view component, std::string_view role) {
    const auto& resources = reader_.directory().component(component).resources;
    const auto found      = resources.find(role);
    if (found == resources.end()) {
        throw ArtifactError(std::string(component) + ": missing resource " + std::string(role));
    }
    (void)host_object(found->second);
    return found->second;
}

HostValues Binder::values(const Binding& binding, std::optional<QType> format) {
    HostValues out;
    out.elements = binding.elements;
    if (binding.parts.empty()) { throw ArtifactError("value Binding is empty"); }
    out.format = format.value_or(reader_.geometry(binding.parts.front().object).format);
    std::uint64_t word_bytes = 0;
    if (out.format == QType::BF16) {
        word_bytes = 2;
    } else if (out.format == QType::FP32 || out.format == QType::INT32) {
        word_bytes = 4;
    } else {
        throw ArtifactError("owning Host values require a direct numeric format");
    }
    const auto total_bytes = checked_mul(out.elements, word_bytes, "Host value bytes");
    if (total_bytes > std::numeric_limits<std::size_t>::max()) {
        throw ArtifactError("Host values exceed size_t");
    }
    out.data.resize(static_cast<std::size_t>(total_bytes));
    std::size_t destination = 0;
    for (const auto& part : binding.parts) {
        const auto& geometry = reader_.geometry(part.object);
        if (geometry.layout != QuantLayout::Contiguous || geometry.format != out.format ||
            part.begin >= part.end || part.end > geometry.elements) {
            throw ArtifactError("Host value Binding has an incompatible representation or range");
        }
        const auto bytes = checked_mul(part.end - part.begin, word_bytes, "value range");
        if (bytes > out.data.size() - destination) {
            throw ArtifactError("Host value coverage exceeds declared size");
        }
        const auto source = checked_mul(part.begin, word_bytes, "value offset");
        auto target = std::span(out.data).subspan(destination, static_cast<std::size_t>(bytes));
        const auto& cached = demands_[part.object.index];
        if (cached.host) {
            std::memcpy(target.data(), cached.host_data.data() + source, target.size());
        } else {
            const auto& object = reader_.directory().tensor(part.object);
            reader_.read_into(checked_add(object.offset, source, "value file offset"), target);
            read_bytes_ = checked_add(read_bytes_, bytes, "Host read bytes");
        }
        destination += target.size();
    }
    if (destination != out.data.size()) {
        throw ArtifactError("Host value coverage is incomplete");
    }
    owned_value_bytes_ = checked_add(owned_value_bytes_, out.data.size(), "owning value bytes");
    return out;
}

MaterializationPlan Binder::finish() && {
    MaterializationPlan plan;
    plan.source            = &reader_;
    plan.object_count      = demands_.size();
    plan.prior_read_bytes  = read_bytes_;
    plan.owned_value_bytes = owned_value_bytes_;
    for (std::size_t i = 0; i < demands_.size(); ++i) {
        auto& demand = demands_[i];
        if (demand.device) {
            const ObjectHandle handle{i};
            const auto& geometry = reader_.geometry(handle);
            const auto offset =
                align_up(plan.device_capacity_bytes, demand.alignment, "device offset");
            plan.device_objects.push_back({handle, offset, geometry.bytes, demand.alignment});
            plan.device_capacity_bytes = checked_add(offset, geometry.bytes, "device capacity");
        }
        if (demand.host) {
            plan.host_objects.push_back({ObjectHandle{i}, std::move(demand.host_data)});
        }
    }
    return plan;
}

} // namespace ninfer::artifact
