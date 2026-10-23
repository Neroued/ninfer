#pragma once

#include "models/common/frontend/resources.h"

namespace ninfer::artifact {
class Binder;
}

namespace ninfer::models::glm5next {

struct FrontendResources {
    models::frontend::TextResources text;
    std::string_view processor_config_json;
    SamplingPreset sampling;
};

// Raw views borrow Definition resource storage; parsed tokenizer and defaults own their values.
void parse_resources(FrontendResources& resources, std::uint32_t embedding_rows);
FrontendResources bind_resources(artifact::Binder& binder, std::uint32_t embedding_rows,
                                 bool vision);

} // namespace ninfer::models::glm5next
