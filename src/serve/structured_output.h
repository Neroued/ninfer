#pragma once
#include "serve/request.h"
#include "serve/request_json.h"

namespace ninfer::serve {
// Chat nests schema metadata under json_schema; Responses and Anthropic use a flat format.
StructuredOutputOptions parse_structured_output(const RequestJson& format, bool nested,
                                                const std::string& param);
} // namespace ninfer::serve
