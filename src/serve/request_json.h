#pragma once

#include <nlohmann/json.hpp>

namespace ninfer::serve {

// Owning JSON representation for request bodies. Object insertion order is part of the
// model-facing representation whenever a nested object is serialized into PromptInput.
using RequestJson = nlohmann::ordered_json;

} // namespace ninfer::serve
