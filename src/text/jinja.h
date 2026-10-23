#pragma once

#include "text/byte_span.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <ctime>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::text {

struct TemplateInputRegion {
    // Descendants inherit whole-value origins; an explicit child region supplies its own origin.
    // Only an explicitly tagged string has exact source-byte coordinates.
    std::string pointer;
    std::uint32_t tag = 0;
};

struct TemplateOutputRegion {
    std::uint32_t tag = 0;
    std::size_t begin = 0;
    std::size_t end   = 0;
    // Absent for an entire transformed region, such as tojson output.
    std::optional<std::size_t> source_offset;
};

struct TemplateBoundaryMapping {
    std::uint32_t tag = 0;
    // Inclusive source-boundary range collapsed by trimming to one output byte position.
    std::size_t source_begin = 0;
    std::size_t source_end   = 0;
    std::size_t offset       = 0;
};

struct TemplateRenderOptions {
    std::time_t timestamp = std::time(nullptr);
    std::function<void()> checkpoint;
    std::span<const TemplateInputRegion> regions;
    // Only engine-supplied token variables opt out of ordinary input-string handling.
    std::span<const std::string> control_variables;
};

struct TemplateOutput {
    std::string text;
    // Ordinary content, independent of optional source-coordinate collection.
    std::vector<ByteSpan> literal_spans;
    std::vector<TemplateOutputRegion> regions;
    std::vector<TemplateBoundaryMapping> boundary_mappings;
};

// The parsed source is immutable. Every render owns its evaluation context.
class JinjaTemplate {
public:
    explicit JinjaTemplate(std::string source, std::string source_name);
    TemplateOutput render(const nlohmann::ordered_json& context,
                          const TemplateRenderOptions& options = {}) const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

} // namespace ninfer::text
