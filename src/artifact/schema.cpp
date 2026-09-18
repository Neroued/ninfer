#include "artifact/schema.h"

#include "core/weight_view.h"

#include <algorithm>
#include <set>
#include <unordered_set>

namespace ninfer::artifact {

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a * b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    if (!alignment) { throw ArtifactError(std::string(label) + ": zero alignment"); }
    return checked_add(value, alignment - 1, label) / alignment * alignment;
}

void require_members(const Json& value, std::initializer_list<std::string_view> required,
                     std::initializer_list<std::string_view> optional, std::string_view label) {
    if (!value.is_object()) { throw ArtifactError(std::string(label) + " must be an object"); }
    for (const auto key : required) {
        if (!value.contains(key)) {
            throw ArtifactError(std::string(label) + ": missing " + std::string(key));
        }
    }
    for (const auto& [key, unused] : value.items()) {
        if (std::find(required.begin(), required.end(), key) == required.end() &&
            std::find(optional.begin(), optional.end(), key) == optional.end()) {
            throw ArtifactError(std::string(label) + ": unknown member " + key);
        }
    }
}

std::string require_id(const Json& value, std::string_view label) {
    if (!value.is_string()) { throw ArtifactError(std::string(label) + " must be a string"); }
    auto result = value.get<std::string>();
    if (result.empty() || result.find('\0') != std::string::npos) {
        throw ArtifactError(std::string(label) + " must be nonempty without NUL");
    }
    return result;
}

std::uint64_t require_u64(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_integer() ||
        (!value.is_number_unsigned() && value.get<std::int64_t>() < 0)) {
        throw ArtifactError(std::string(label) + " must be a nonnegative integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && !result) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

const std::string& object_id(const Object& object) {
    return std::visit([](const auto& item) -> const std::string& { return item.id; }, object);
}

std::uint64_t object_offset(const Object& object) {
    return std::visit([](const auto& item) { return item.offset; }, object);
}

std::uint64_t object_bytes(const Object& object) {
    return std::visit([](const auto& item) { return item.bytes; }, object);
}

const Object& Directory::object(ObjectHandle handle) const {
    if (handle.index >= objects.size()) { throw ArtifactError("invalid object handle"); }
    return objects[handle.index];
}

const TensorObject& Directory::tensor(ObjectHandle handle) const {
    const auto* result = std::get_if<TensorObject>(&object(handle));
    if (!result) { throw ArtifactError("object does not reference a tensor"); }
    return *result;
}

const Component& Directory::component(std::string_view name) const {
    const auto found = components.find(name);
    if (found == components.end()) {
        throw ArtifactError("missing component " + std::string(name));
    }
    return found->second;
}

Json parse_json(std::string_view text, std::string_view label) {
    std::vector<std::unordered_set<std::string>> member_stack;
    const auto callback = [&](int, Json::parse_event_t event, Json& parsed) {
        if (event == Json::parse_event_t::object_start) {
            member_stack.emplace_back();
        } else if (event == Json::parse_event_t::key) {
            const auto& key = parsed.get_ref<const std::string&>();
            if (!member_stack.back().insert(key).second) {
                throw ArtifactError(std::string(label) + ": duplicate JSON member " + key);
            }
        } else if (event == Json::parse_event_t::object_end) {
            member_stack.pop_back();
        }
        return true;
    };
    try {
        return Json::parse(text.begin(), text.end(), callback);
    } catch (const Json::exception& error) {
        throw ArtifactError(std::string(label) + ": " + error.what());
    }
}

namespace {

Shape parse_shape(const Json& value, std::string_view label) {
    if (!value.is_array() || value.size() > 16) {
        throw ArtifactError(std::string(label) + ": shape must have rank 0 through 16");
    }
    Shape out;
    std::uint64_t elements = 1;
    for (const auto& item : value) {
        const auto dim = require_u64(item, label, true);
        elements       = checked_mul(elements, dim, label);
        out.push_back(dim);
    }
    return out;
}

ObjectHandle reference(const Directory& directory, const Json& value, std::string_view label,
                       bool tensor) {
    const auto name  = require_id(value, label);
    const auto found = directory.object_index.find(name);
    if (found == directory.object_index.end()) {
        throw ArtifactError(std::string(label) + ": missing object " + name);
    }
    if (std::holds_alternative<TensorObject>(directory.object(found->second)) != tensor) {
        throw ArtifactError(std::string(label) + ": wrong object kind for " + name);
    }
    return found->second;
}

Binding parse_binding(const Directory& directory, const Json& value, std::string_view label) {
    if (!value.is_object()) {
        throw ArtifactError(std::string(label) + ": expected Binding object");
    }
    Binding out;
    if (value.size() == 1 && value.contains("object")) {
        const auto handle  = reference(directory, value.at("object"), label, true);
        const auto& object = directory.tensor(handle);
        out.whole_object   = true;
        out.elements       = weight_element_count(object.shape);
        out.parts.push_back({handle, 0, out.elements});
        return out;
    }
    require_members(value, {"parts"}, {}, label);
    const auto& parts = value.at("parts");
    if (!parts.is_array() || parts.empty()) {
        throw ArtifactError(std::string(label) + ": parts must be a nonempty array");
    }
    for (const auto& part : parts) {
        require_members(part, {"object", "range"}, {}, label);
        const auto handle  = reference(directory, part.at("object"), label, true);
        const auto& bounds = part.at("range");
        if (!bounds.is_array() || bounds.size() != 2) {
            throw ArtifactError(std::string(label) + ": range must contain two integers");
        }
        const auto begin = require_u64(bounds[0], label);
        const auto end   = require_u64(bounds[1], label);
        if (begin >= end || end > weight_element_count(directory.tensor(handle).shape)) {
            throw ArtifactError(std::string(label) + ": Part exceeds source logical elements");
        }
        out.elements = checked_add(out.elements, end - begin, label);
        out.parts.push_back({handle, begin, end});
    }
    return out;
}

void parse_files(Directory& out, const Json& files, std::string_view entry_name) {
    if (!files.is_array() || files.empty()) {
        throw ArtifactError("files must be a nonempty array");
    }
    std::set<std::string, std::less<>> names{std::string(entry_name)};
    for (const auto& value : files) {
        require_members(value, {"path", "payload_bytes"}, {}, "file");
        FileRecord file;
        if (out.files.empty()) {
            if (!value.at("path").is_null()) { throw ArtifactError("files[0].path must be null"); }
        } else {
            file.path        = require_id(value.at("path"), "file path");
            const auto& path = *file.path;
            if (path == "." || path == ".." || path.find('/') != std::string::npos ||
                path.find('\\') != std::string::npos || !names.insert(path).second) {
                throw ArtifactError("invalid or duplicate sibling filename: " + path);
            }
        }
        file.payload_bytes = require_u64(value.at("payload_bytes"), "file payload bytes", true);
        file.logical_begin = out.payload_bytes;
        out.payload_bytes =
            checked_add(out.payload_bytes, file.payload_bytes, "total payload bytes");
        out.files.push_back(std::move(file));
    }
}

void parse_objects(Directory& out, const Json& objects) {
    if (!objects.is_array() || objects.empty()) {
        throw ArtifactError("objects must be a nonempty array");
    }
    std::uint64_t previous_end = 0;
    for (const auto& value : objects) {
        if (!value.is_object() || !value.contains("kind")) {
            throw ArtifactError("invalid object descriptor");
        }
        const auto kind = require_id(value.at("kind"), "object kind");
        if (kind == "tensor") {
            require_members(value, {"id", "kind", "shape", "format", "layout", "offset", "bytes"},
                            {}, "tensor");
        } else if (kind == "resource") {
            require_members(value, {"id", "kind", "encoding", "offset", "bytes"}, {}, "resource");
        } else {
            throw ArtifactError("unknown object kind " + kind);
        }
        auto id           = require_id(value.at("id"), "object id");
        const auto offset = require_u64(value.at("offset"), id);
        const auto bytes  = require_u64(value.at("bytes"), id, true);
        const auto end    = checked_add(offset, bytes, id);
        if (offset < previous_end || end > out.payload_bytes) {
            throw ArtifactError(id + ": object overlaps, is out of order or exceeds payload");
        }
        if (!out.object_index.emplace(id, ObjectHandle{out.objects.size()}).second) {
            throw ArtifactError("duplicate object id " + id);
        }
        previous_end = end;
        if (kind == "tensor") {
            out.objects.emplace_back(TensorObject{
                id, parse_shape(value.at("shape"), id), require_id(value.at("format"), id),
                require_id(value.at("layout"), id), offset, bytes});
        } else {
            out.objects.emplace_back(
                ResourceObject{id, require_id(value.at("encoding"), id), offset, bytes});
        }
    }
}

void parse_components(Directory& out, const Json& values) {
    if (!values.is_object() || !values.contains("text")) {
        throw ArtifactError("components must contain text");
    }
    for (const auto& [name, value] : values.items()) {
        (void)require_id(Json(name), "component id");
        require_members(value, {"config"}, {"target", "resources", "proposal"}, name);
        Component component;
        component.config = value.at("config");
        if (!component.config.is_object()) {
            throw ArtifactError(name + ": config must be an object");
        }
        if (value.contains("target")) {
            component.target = require_id(value.at("target"), name + " target");
            if (!values.contains(*component.target)) {
                throw ArtifactError(name + ": missing target " + *component.target);
            }
        }
        if (value.contains("resources")) {
            const auto& resources = value.at("resources");
            if (!resources.is_object()) {
                throw ArtifactError(name + ": resources must be an object");
            }
            for (const auto& [role, ref] : resources.items()) {
                (void)require_id(Json(role), "resource role");
                component.resources.emplace(role, reference(out, ref, name + "/" + role, false));
            }
        }
        if (value.contains("proposal")) {
            if (name != "text") { throw ArtifactError("only text may declare proposal"); }
            const auto& p = value.at("proposal");
            require_members(p, {"domain"}, {"rows"}, "proposal");
            const auto domain = require_id(p.at("domain"), "proposal domain");
            if (domain == "full" && !p.contains("rows")) {
                component.proposal = Proposal{};
            } else if (domain == "indexed" && p.contains("rows")) {
                component.proposal =
                    Proposal{true, require_u64(p.at("rows"), "proposal rows", true)};
            } else {
                throw ArtifactError(
                    "proposal must be full without rows or indexed with positive rows");
            }
        }
        out.components.emplace(name, std::move(component));
    }
}

void parse_uses(Directory& out, const Json& values) {
    if (!values.is_array()) { throw ArtifactError("uses must be an array"); }
    for (const auto& value : values) {
        require_members(value, {"parameter", "input"}, {"activation_policy", "auxiliaries"}, "Use");
        Use use;
        use.parameter    = require_id(value.at("parameter"), "Use parameter");
        use.input        = require_id(value.at("input"), "Use input");
        const auto label = use.parameter + "@" + use.input;
        if (!out.bindings.contains(use.parameter)) {
            throw ArtifactError(label + ": missing parameter");
        }
        if (value.contains("activation_policy")) {
            const auto policy = require_id(value.at("activation_policy"), label);
            if (policy == "A16Only") {
                use.activation_policy = ActivationPolicy::A16Only;
            } else if (policy == "AllowA8") {
                use.activation_policy = ActivationPolicy::AllowA8;
            } else if (policy == "AllowA4") {
                use.activation_policy = ActivationPolicy::AllowA4;
            } else {
                throw ArtifactError(label + ": unknown activation policy " + policy);
            }
        }
        if (value.contains("auxiliaries")) {
            const auto& auxiliaries = value.at("auxiliaries");
            if (!auxiliaries.is_object()) {
                throw ArtifactError(label + ": auxiliaries must be an object");
            }
            for (const auto& [role, binding] : auxiliaries.items()) {
                (void)require_id(Json(role), "auxiliary role");
                use.auxiliaries.emplace(role, parse_binding(out, binding, label + "/" + role));
            }
        }
        const auto key = std::pair{use.parameter, use.input};
        if (!out.uses.emplace(key, std::move(use)).second) {
            throw ArtifactError(label + ": duplicate Use");
        }
    }
}

} // namespace

Directory parse_directory(const Json& root, std::string_view entry_name) {
    require_members(root, {"components", "objects", "bindings", "uses", "files"},
                    {"metadata", "provenance"}, "directory");
    Directory out;
    parse_files(out, root.at("files"), entry_name);
    parse_objects(out, root.at("objects"));
    parse_components(out, root.at("components"));
    const auto& bindings = root.at("bindings");
    if (!bindings.is_object()) { throw ArtifactError("bindings must be an object"); }
    for (const auto& [name, value] : bindings.items()) {
        (void)require_id(Json(name), "parameter name");
        out.bindings.emplace(name, parse_binding(out, value, name));
    }
    parse_uses(out, root.at("uses"));
    if (root.contains("metadata")) { out.metadata = root.at("metadata"); }
    if (root.contains("provenance")) { out.provenance = root.at("provenance"); }
    if (!out.metadata.is_object() || !out.provenance.is_object()) {
        throw ArtifactError("metadata and provenance must be objects");
    }
    if (out.metadata.contains("name")) {
        (void)require_id(out.metadata.at("name"), "metadata.name");
    }
    return out;
}

} // namespace ninfer::artifact
