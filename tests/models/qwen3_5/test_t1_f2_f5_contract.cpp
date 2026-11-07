// T1 / F2 + F5: default-chunk single authority, mixed/pack sizing, and the mixed descriptors.
// No GPU needed. Source checks locate files from this file's path.
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/planning.h"
#include "models/qwen3_5/prefill_chunk.h"
#include "models/qwen3_5/work.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

namespace {

namespace q  = ninfer::models::qwen3_5;
namespace x  = ninfer::models::qwen3_5::execution;
namespace fs = std::filesystem;
using ninfer::models::WeightPlacement;

int failures = 0;

void expect(bool ok, std::string_view msg) {
    if (ok) return;
    ++failures;
    std::cerr << "FAIL: " << msg << '\n';
}

std::string slurp(const fs::path& p) {
    std::ifstream in(p);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

template <class F> bool throws_invalid(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return true; } catch (...) { return false; }
    return false;
}

// F5 descriptors: compile-time contract.
static_assert(std::is_same_v<decltype(x::StepRows::token_offset), std::uint32_t>);
static_assert(std::is_same_v<decltype(x::StepRows::width), std::uint32_t>);
static_assert(std::is_same_v<decltype(x::StepRows::rows), std::uint32_t>);
static_assert(std::is_same_v<decltype(x::StepRows::row_offset), std::uint32_t>);
static_assert(std::is_constructible_v<x::TextStateUpdate, x::MixedUpdate>);
static_assert(std::is_same_v<decltype(q::MixedWork::prefill), q::ForwardWork>);
static_assert(std::is_same_v<decltype(q::MixedWork::decode), q::DecodeWork>);
static_assert(std::is_same_v<decltype(q::MixedResult::prefill), q::ForwardResult>);
static_assert(std::is_same_v<decltype(q::MixedResult::decode), q::GeneratedResult>);

} // namespace

int main() {
    const fs::path root = fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path();

    // F2 acceptance 1/2: one definition site; default valid and within max_context.
    {
        int defs = 0;
        for (const auto& e : fs::recursive_directory_iterator(root / "src")) {
            if (!e.is_regular_file()) continue;
            const auto ext = e.path().extension();
            if (ext != ".h" && ext != ".cpp" && ext != ".cu" && ext != ".cuh") continue;
            const auto text = slurp(e.path());
            std::istringstream lines(text);
            for (std::string line; std::getline(lines, line);)
                if (line.find("constexpr") != std::string::npos &&
                    line.find("kResidentPrefillChunk") != std::string::npos)
                    ++defs;
        }
        expect(defs == 1, "kResidentPrefillChunk is defined in exactly one place");
        ninfer::EngineOptions o;
        o.max_context = 262144;
        const auto c  = q::resolve_prefill_chunk(o, WeightPlacement::Resident);
        expect(c == q::kResidentPrefillChunk && c % 128 == 0 && c <= o.max_context,
               "resident default is valid and equals the single constant");
        expect(q::resolve_prefill_chunk(o, WeightPlacement::HostExperts) == 4096,
               "host-experts default stays 4096");
        o.prefill_chunk = 2048;
        expect(q::resolve_prefill_chunk(o, WeightPlacement::Resident) == 2048, "explicit 2048 wins");
        o.prefill_chunk = 1000;
        expect(throws_invalid([&] { (void)q::resolve_prefill_chunk(o, WeightPlacement::Resident); }),
               "non-multiple of 128 rejected");
    }

    // F2 acceptance 3: moe real test derives its default expectation from the constants.
    {
        const auto t = slurp(root / "tests/models/qwen3_5/test_engine_moe_real.cpp");
        expect(t.find("kResidentPrefillChunk") != std::string::npos &&
                   t.find("prefill_chunk   = 2048") != std::string::npos,
               "moe real test keeps explicit 2048 check and uses the default constant");
    }

    // F2 acceptance 4: startup sites no longer hard-code a numeric default chunk.
    for (const char* rel : {"src/models/qwen3_5/startup.cpp",
                            "src/models/qwen3_5/program/planning/startup.cpp"}) {
        const auto t = slurp(root / rel);
        expect(!t.empty(), std::string("readable: ") + rel);
        expect(t.find("? 4096") == std::string::npos && t.find(": 1024") == std::string::npos &&
                   t.find("= 1024") == std::string::npos,
               std::string("no hard-coded default chunk in ") + rel);
    }

    // F5: pack and mixed sizing.
    {
        const auto pack = q::resolve_prefill_pack_tokens(q::kResidentPrefillChunk, 262144);
        expect(pack % 128 == 0 && pack >= q::kResidentPrefillChunk, "pack capacity valid");
        expect(q::resolve_mixed_tokens(pack, 0, 3) == 0, "zero decode rows -> not mixed");
        expect(q::resolve_mixed_tokens(pack, 6, 3) == pack + 6u * 4u, "mixed = pack + rows*(window+1)");
        expect(throws_invalid([&] { (void)q::resolve_mixed_tokens(100, 6, 3); }),
               "unaligned pack tokens rejected");
        x::StepRows s{};
        expect(s.rows == 0 && s.width == 0, "StepRows value-initializes");
    }

    if (failures == 0) std::cout << "ok\n";
    return failures == 0 ? 0 : 1;
}
