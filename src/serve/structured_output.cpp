#include "serve/structured_output.h"
#include "text/structured_output.h"

namespace ninfer::serve {
StructuredOutputOptions parse_structured_output(const RequestJson& format, bool nested,
                                                const std::string& param) {
    const auto fail = [&](const std::string& message) -> void {
        throw ApiException(
            ApiError{.message = message, .param = param, .code = "invalid_response_format"});
    };
    if (!format.is_object() || !format.contains("type") || !format.at("type").is_string()) {
        fail(param + " must contain a string type");
    }
    const auto type = format.at("type").get<std::string>();
    StructuredOutputOptions options;
    if (type == "text" || type == "json_object") {
        if (format.size() != 1) { fail("unexpected members in " + param); }
        options.kind =
            type == "text" ? StructuredOutputKind::None : StructuredOutputKind::JsonObject;
    } else if (type == "json_schema") {
        const auto& spec =
            nested && format.contains("json_schema") ? format.at("json_schema") : format;
        if (nested && (format.size() != 2 || !format.contains("json_schema"))) {
            fail("json_schema requires a json_schema object");
        }
        if (!spec.is_object() || !spec.contains("schema") ||
            (!spec.at("schema").is_object() && !spec.at("schema").is_boolean())) {
            fail("json_schema requires an object or boolean schema");
        }
        for (const auto& [key, value] : spec.items()) {
            if (key == "schema" || (!nested && key == "type")) { continue; }
            if ((key == "name" || key == "description") && value.is_string()) { continue; }
            if (key == "strict" && (value.is_boolean() || value.is_null())) { continue; }
            fail("unsupported or invalid schema format member: " + key);
        }
        options.kind   = StructuredOutputKind::JsonSchema;
        options.schema = spec.at("schema").dump();
    } else {
        fail("unsupported response format type: " + type);
    }
    try {
        text::validate_structured_output(options);
    } catch (const std::invalid_argument& e) { fail(e.what()); }
    return options;
}
} // namespace ninfer::serve
