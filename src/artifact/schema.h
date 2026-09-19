#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::artifact {

class ArtifactError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

using Json  = nlohmann::json;
using Shape = std::vector<std::uint64_t>;

struct ObjectHandle {
    std::size_t index                                  = std::numeric_limits<std::size_t>::max();
    friend bool operator==(ObjectHandle, ObjectHandle) = default;
};

struct TensorObject {
    std::string id;
    Shape shape;
    std::string format;
    std::string layout;
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
    // Source matrices this plane was assembled from, each with its own NVFP4 divisor. Absent means
    // one, which is every plane that is not a stack of separately quantised matrices.
    std::uint64_t divisors = 1;
};

struct ResourceObject {
    std::string id;
    std::string encoding;
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
};

using Object = std::variant<TensorObject, ResourceObject>;
[[nodiscard]] const std::string& object_id(const Object& object);
[[nodiscard]] std::uint64_t object_offset(const Object& object);
[[nodiscard]] std::uint64_t object_bytes(const Object& object);

struct Part {
    ObjectHandle object;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

struct Binding {
    bool whole_object = false;
    std::vector<Part> parts;
    std::uint64_t elements = 0;
};

enum class ActivationPolicy { A16Only, AllowA8, AllowA4 };

struct Use {
    std::string parameter;
    std::string input;
    std::optional<ActivationPolicy> activation_policy;
    std::map<std::string, Binding, std::less<>> auxiliaries;
};

struct Proposal {
    bool indexed       = false;
    std::uint64_t rows = 0;
};

struct Component {
    Json config;
    std::optional<std::string> target;
    std::map<std::string, ObjectHandle, std::less<>> resources;
    std::optional<Proposal> proposal;
};

struct FileRecord {
    std::optional<std::string> path;
    std::uint64_t payload_bytes = 0;
    std::uint64_t logical_begin = 0;
};

struct Directory {
    std::map<std::string, Component, std::less<>> components;
    std::vector<Object> objects;
    std::map<std::string, ObjectHandle, std::less<>> object_index;
    std::map<std::string, Binding, std::less<>> bindings;
    std::map<std::pair<std::string, std::string>, Use> uses;
    std::vector<FileRecord> files;
    std::uint64_t payload_bytes = 0;
    Json metadata               = Json::object();
    Json provenance             = Json::object();

    [[nodiscard]] const Object& object(ObjectHandle handle) const;
    [[nodiscard]] const TensorObject& tensor(ObjectHandle handle) const;
    [[nodiscard]] const Component& component(std::string_view name) const;
};

[[nodiscard]] Json parse_json(std::string_view text, std::string_view label);
[[nodiscard]] Directory parse_directory(const Json& root, std::string_view entry_name);

// Shared cold-data primitives. Architecture config parsers use these without moving their
// field sets or mathematical constraints into the container schema.
void require_members(const Json& value, std::initializer_list<std::string_view> required,
                     std::initializer_list<std::string_view> optional, std::string_view label);
[[nodiscard]] std::string require_id(const Json& value, std::string_view label);
[[nodiscard]] std::uint64_t require_u64(const Json& value, std::string_view label,
                                        bool positive = false);
[[nodiscard]] std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label);
[[nodiscard]] std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, std::string_view label);
[[nodiscard]] std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment,
                                     std::string_view label);

} // namespace ninfer::artifact
