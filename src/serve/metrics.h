#pragma once

#include "serve/observations.h"

namespace ninfer::serve {

// Prometheus formatting over the same post-warmup observations used by the JSON API.
class Metrics {
public:
    void configure(std::string model, const EngineOptions&, const StartupReport&,
                   double start_seconds);
    [[nodiscard]] std::string render(const EngineSnapshot&, const ServiceSnapshot&,
                                     bool ready) const;
private:
    std::string model_;
    EngineOptions options_;
    StartupReport startup_;
    double start_seconds_ = 0;
};

} // namespace ninfer::serve
