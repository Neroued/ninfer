// T1 / F2: public contract of the Qwen3.5 prefill_chunk default resolution. No GPU needed.
#include "models/qwen3_5/prefill_chunk.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

namespace q = ninfer::models::qwen3_5;
using ninfer::models::WeightPlacement;

int failures = 0;

void expect(bool ok, std::string_view msg) {
    if (ok) return;
    ++failures;
    std::cerr << "FAIL: " << msg << '\n';
}

ninfer::EngineOptions options(std::uint32_t max_context, std::uint32_t chunk = 0) {
    ninfer::EngineOptions o;
    o.max_context   = max_context;
    o.prefill_chunk = chunk;
    return o;
}

bool throws_invalid(const ninfer::EngineOptions& o, WeightPlacement p) {
    try {
        (void)q::resolve_prefill_chunk(o, p);
    } catch (const std::invalid_argument&) { return true; } catch (...) { return false; }
    return false;
}

} // namespace

int main() {
    // Defaults are valid chunks themselves.
    expect(q::kResidentPrefillChunk % 128 == 0 && q::kResidentPrefillChunk > 0,
           "resident default is a positive multiple of 128");
    expect(q::kHostExpertsPrefillChunk == 4096, "HostExperts default stays 4096");

    // Unset chunk at the judge's max_context resolves to the placement default.
    expect(q::resolve_prefill_chunk(options(262144), WeightPlacement::Resident) ==
               q::kResidentPrefillChunk,
           "resident default at max_context=262144");
    expect(q::resolve_prefill_chunk(options(262144), WeightPlacement::HostExperts) ==
               q::kHostExpertsPrefillChunk,
           "host-experts default at max_context=262144");

    // Result is clamped to max_context.
    expect(q::resolve_prefill_chunk(options(128), WeightPlacement::Resident) == 128,
           "resident default clamps to max_context=128");
    expect(q::resolve_prefill_chunk(options(256), WeightPlacement::HostExperts) == 256,
           "host-experts default clamps to max_context=256");

    // Explicit value wins over placement default.
    expect(q::resolve_prefill_chunk(options(262144, 2048), WeightPlacement::Resident) == 2048,
           "explicit 2048 wins (resident)");
    expect(q::resolve_prefill_chunk(options(262144, 1024), WeightPlacement::HostExperts) == 1024,
           "explicit 1024 wins (host experts)");

    // Non-multiples of 128 are rejected with std::invalid_argument.
    expect(throws_invalid(options(262144, 1000), WeightPlacement::Resident),
           "explicit 1000 rejected");
    expect(throws_invalid(options(262144, 129), WeightPlacement::HostExperts),
           "explicit 129 rejected");

    if (failures == 0) std::cout << "ok\n";
    return failures == 0 ? 0 : 1;
}
