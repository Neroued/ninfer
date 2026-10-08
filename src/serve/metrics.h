#pragma once

#include "serve/generation_service.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>

namespace ninfer::serve {

// Fixed-size request aggregates. Runtime counters remain owned and published by Engine.
class Metrics {
public:
    void configure(std::string model, const EngineOptions& options, const MemorySummary& memory,
                   RuntimeStats baseline);
    void first_token(const GenerationFirstTokenObservation& observation);
    void done(const GenerationOutcome& outcome);
    void rejected();
    void failed(bool cancelled);
    void response_failed();
    [[nodiscard]] std::string render(const RuntimeStats& runtime, bool ready) const;

private:
    struct Histogram {
        // Disjoint bins while recording; cumulative buckets are produced only when scraped.
        std::array<std::uint64_t, 21> bins{};
        double sum = 0.0;
        void observe(double seconds);
    };

    struct Requests {
        std::uint64_t completed         = 0;
        std::uint64_t cancelled         = 0;
        std::uint64_t failed            = 0;
        std::uint64_t rejected          = 0;
        std::uint64_t response_failures = 0;
        std::array<std::uint64_t, 3> constraint_outcomes{};
        std::array<std::uint64_t, 3> constraint_cache{};
        double constraint_prepare_seconds     = 0;
        double constraint_mask_seconds        = 0;
        double constraint_matcher_seconds     = 0;
        std::uint64_t constraint_positions    = 0;
        std::uint64_t constraint_upload_bytes = 0;
        Histogram ttft;
        Histogram duration;
        Histogram queue;
    };

    std::string model_;
    EngineOptions options_;
    MemorySummary memory_;
    RuntimeStats baseline_;
    double start_seconds_ = 0.0;
    mutable std::mutex mutex_;
    Requests requests_;
};

} // namespace ninfer::serve
