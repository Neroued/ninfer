#include "runtime/engine/model_instance.h"

#include <iostream>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using ninfer::EngineOptions;
    using ninfer::kMaximumPreparedPromptCacheCandidatesPerRequest;
    using ninfer::runtime::normalize_engine_options;

    int failures = 0;

    // A single request can produce up to kMaximumPreparedPromptCacheCandidatesPerRequest distinct
    // shared-prefix candidates (frontend.cpp's opportunities.reserve(7U): four explicit markers
    // plus the engine's tool/leading-instruction/full-prompt automatic candidates). The default
    // Engine-wide shared catalog must be able to hold at least one request's own candidates even
    // at the smallest concurrency, or ordinary DefaultAutomatic-evidence traffic starts losing
    // cache hits to its own prior turns as soon as the catalog fills.
    for (const std::uint32_t concurrency : {1U, 2U, 8U}) {
        EngineOptions options;
        options.max_concurrency = concurrency;
        const EngineOptions normalized = normalize_engine_options(options);
        const std::uint32_t default_shared_prefixes = *normalized.context_cache.max_shared_prefixes;
        failures += check(
            default_shared_prefixes >=
                static_cast<std::uint32_t>(kMaximumPreparedPromptCacheCandidatesPerRequest),
            "default shared-prefix catalog capacity is smaller than one request's own candidate ceiling");
        failures += check(default_shared_prefixes >= concurrency,
                          "default shared-prefix catalog capacity did not cover active concurrency");
    }

    // An explicit override is still respected verbatim, including a deliberately small value.
    {
        EngineOptions options;
        options.max_concurrency               = 1;
        options.context_cache.max_shared_prefixes = 1;
        const EngineOptions normalized = normalize_engine_options(options);
        failures += check(*normalized.context_cache.max_shared_prefixes == 1,
                          "explicit max_shared_prefixes override was not preserved");
    }

    // A disabled context cache still normalizes to a root-only zero capacity.
    {
        EngineOptions options;
        options.max_concurrency        = 1;
        options.context_cache.enabled  = false;
        const EngineOptions normalized = normalize_engine_options(options);
        failures += check(*normalized.context_cache.max_shared_prefixes == 0,
                          "disabled context cache did not normalize shared-prefix capacity to zero");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
