#pragma once

#include "utils.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace jinja {

namespace unicode {
enum class Case;
}

// Optional origins locate input regions. Literal bytes remain ordinary content even when a
// transformation loses that location; neither property changes Jinja string equality or text.
struct string_part {
    std::uint32_t origin = 0;
    std::string val;
    std::optional<std::size_t> source_offset = 0;
    bool literal                             = false;
};

struct string {
    std::vector<string_part> parts;
    string() = default;

    string(const std::string& value, std::uint32_t origin = 0, bool literal = false)
        : parts{{origin, value, 0, literal}} {}

    string(int value) : string(std::to_string(value)) {}

    string(double value) : string(std::to_string(value)) {}

    void tag(std::uint32_t origin, bool exact = true);
    std::string str() const;
    std::size_t length() const;
    std::size_t byte_size() const;
    void hash_update(hasher& hash) const noexcept;
    bool is_uppercase() const;
    bool is_lowercase() const;
    string& append(const string& other);
    string cut_bytes(std::size_t begin, std::size_t end) const;
    string slice(std::optional<std::int64_t> start, std::optional<std::int64_t> stop,
                 std::int64_t step = 1) const;
    std::vector<string> split(const std::optional<std::string>& separator, int64_t maxsplit,
                              bool reverse) const;
    string uppercase() const;
    string lowercase() const;
    string capitalize() const;
    string titlecase() const;
    string strip(bool left, bool right,
                 std::optional<const std::string_view> chars = std::nullopt) const;

private:
    string map_case(unicode::Case mode) const;
};

} // namespace jinja
