#pragma once

#include "decision/decision.h"

namespace ninfer::serve {

enum class SystemOneType { Choice, Noul, Score, Box };

struct SystemOneQuestion {
    std::string id;
    SystemOneType type;
    decision::Json criteria;
    std::size_t first_field = 0;
    std::string aggregate   = "mode";
};

struct SystemOneRequest {
    decision::Request decision;
    std::vector<SystemOneQuestion> questions;
    std::vector<std::string> images;
};

// Pure host protocol translation. No HTTP self-call, runtime, model or GPU allocation.
[[nodiscard]] SystemOneRequest parse_systemone_request(const decision::Json& body);
[[nodiscard]] decision::Json format_systemone_response(const SystemOneRequest& request,
                                                       const decision::Json& result,
                                                       const std::string& served_model);

} // namespace ninfer::serve
