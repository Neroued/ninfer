// T6 / F5: public contract of packed prefill sizing and segment descriptors. No GPU needed.
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/planning.h"
#include "models/qwen3_5/prefill_chunk.h"

#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <variant>

namespace {

namespace q = ninfer::models::qwen3_5;
namespace x = ninfer::models::qwen3_5::execution;

int failures = 0;

void expect(bool ok, std::string_view msg) {
    if (ok) return;
    ++failures;
    std::cerr << "FAIL: " << msg << '\n';
}

bool throws_invalid(std::uint32_t chunk, std::uint32_t ctx) {
    try {
        (void)q::resolve_prefill_pack_tokens(chunk, ctx);
    } catch (const std::invalid_argument&) { return true; } catch (...) { return false; }
    return false;
}

static_assert(std::is_same_v<decltype(x::SequenceSegment::source_offset), std::int64_t>);
static_assert(std::is_same_v<decltype(x::SequenceSegment::destination_offset), std::int64_t>);
static_assert(std::is_constructible_v<x::TextStateUpdate, x::PackedSequenceUpdate>);

} // namespace

int main() {
    expect(q::kPrefillPackChunks >= 1, "kPrefillPackChunks is positive");

    // Pack capacity: multiple of 128, at least one chunk, never above the aligned context.
    for (std::uint32_t chunk : {128u, 512u, 2048u, 4096u}) {
        for (std::uint32_t ctx : {chunk, 262144u}) {
            const auto p = q::resolve_prefill_pack_tokens(chunk, ctx);
            expect(p % 128 == 0, "pack tokens is a multiple of 128");
            expect(p >= chunk, "pack tokens holds at least one chunk");
            if (ctx >= chunk) expect(p <= (ctx + 127) / 128 * 128, "pack tokens bounded by context");
        }
    }
    // A pack holds more than one chunk when the context allows it (amortisation is possible).
    expect(q::resolve_prefill_pack_tokens(2048, 262144) > 2048, "pack exceeds one chunk at 262144");
    // A context of exactly one chunk gives a single-chunk pack.
    expect(q::resolve_prefill_pack_tokens(128, 128) == 128, "pack == chunk when context == chunk");

    expect(throws_invalid(0, 262144), "chunk 0 rejected");
    expect(throws_invalid(1000, 262144), "chunk 1000 rejected");

    // Segment descriptors: consecutive segments of unequal lengths (incl. 1) are expressible.
    const x::SequenceSegment segs[] = {{0, 0, 1, 0, 64}, {1, 1, 300, 128, 192}, {2, 301, 2048, 256, 320}};
    x::PackedSequenceUpdate up{std::span<const x::SequenceSegment>(segs)};
    x::TextStateUpdate u = up;
    expect(std::holds_alternative<x::PackedSequenceUpdate>(u), "variant holds packed update");
    expect(std::get<x::PackedSequenceUpdate>(u).segments.size() == 3, "three segments kept");

    if (failures == 0) std::cout << "ok\n";
    return failures == 0 ? 0 : 1;
}
