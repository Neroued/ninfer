#pragma once

// Structured-output request and observation types (constrained decoding). The definitions are
// taken from Neroued/ninfer 77c2e5e2 (include/ninfer/types.h and
// include/ninfer/observability/execution.h) and kept in a capy-flash-owned header so that
// types.h only gains an include. Nothing in the engine consumes them yet: the text library
// (src/text/grammar.h) is their only user until the frontend/engine hooks land.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ninfer {

enum class OutputConstraintKind : std::uint8_t { Grammar, JsonObject, JsonSchema, Choice, Regex };

// Constrains generated content; Chat reasoning retains the model's framing. Source is owning
// GBNF, JSON Schema or regex text. Choice owns literal alternatives; JsonObject has no payload.
struct OutputConstraint {
    OutputConstraintKind kind = OutputConstraintKind::Grammar;
    std::string source;
    std::vector<std::string> choices;

    [[nodiscard]] static OutputConstraint grammar(std::string source) {
        return {OutputConstraintKind::Grammar, std::move(source), {}};
    }

    [[nodiscard]] static OutputConstraint json_object() {
        return {OutputConstraintKind::JsonObject, {}, {}};
    }

    [[nodiscard]] static OutputConstraint json_schema(std::string source) {
        return {OutputConstraintKind::JsonSchema, std::move(source), {}};
    }

    [[nodiscard]] static OutputConstraint choice(std::vector<std::string> values) {
        return {OutputConstraintKind::Choice, {}, std::move(values)};
    }

    [[nodiscard]] static OutputConstraint regex(std::string pattern) {
        return {OutputConstraintKind::Regex, std::move(pattern), {}};
    }

    bool operator==(const OutputConstraint&) const = default;
};

// capy-flash: limits that keep one client's constraint from stopping the server, stalling the
// other requests or exhausting host memory. Input limits are checked before compilation (and
// already in the HTTP layer); the compilation runs under a deadline with bounded concurrency; the
// matcher work of a generating request runs under a per-round limit, a cumulative budget of
// request_budget_ms + round_allowance_ms per round and a memory bound. Time and memory limits of 0
// disable that check.
struct OutputConstraintLimits {
    // JSON Schema text (as serialized), GBNF or regex text, or the total bytes of the choices.
    std::size_t max_bytes = 256U * 1024U;
    // JSON nesting of the schema document (objects and arrays); group nesting of GBNF and regex.
    std::uint32_t max_depth = 256;
    // Members of one JSON object or array in the schema (properties, enum values, branches...),
    // choice entries, GBNF rules.
    std::uint32_t max_items = 4096;
    // Wall time of one compilation, from the moment it holds a compilation slot.
    std::uint32_t compile_timeout_ms = 5000;
    // Compilations that may run at the same time (cache hits do not take a slot).
    std::uint32_t compile_concurrency = 2;
    // Longest wait for a compilation slot (first come, first served); then 429 "busy".
    std::uint32_t compile_wait_ms = 30000;
    // Compiled grammar cache of the model (LRU).
    std::size_t cache_bytes = 64U * 1024U * 1024U;
    // Matcher CPU time of one request in one decode round (masks of all positions, or the
    // commit's accept), on the calling thread; the wall clock is checked at 5x as a safety net.
    std::uint32_t round_budget_ms = 20;
    // Cumulative matcher CPU time of one request: request_budget_ms plus round_allowance_ms for
    // every round it has run. 5 ms per round: a legitimate domain-name `pattern` measured 3.4 ms
    // per round with every core busy (Flash-Next, 48 busy loops), where 2 ms failed it.
    std::uint32_t request_budget_ms  = 1000;
    std::uint32_t round_allowance_ms = 5;
    // Memory of one request's matcher history (it grows with the output, quadratically for an
    // ambiguous grammar). 0 disables the check.
    std::size_t matcher_bytes = 512U * 1024U * 1024U;
    // JSON Schema in XGrammar's strict mode (its default, as in vLLM/SGLang): objects without
    // additionalProperties and arrays without items/prefixItems admit nothing undeclared. false =
    // literal JSON Schema (open objects and arrays; an extra key is costly for the matcher).
    bool json_strict = true;

    bool operator==(const OutputConstraintLimits&) const = default;
};

// Largest accepted max_depth: deeper documents could exhaust the stack of the recursive
// converters, whatever the configuration.
inline constexpr std::uint32_t kMaximumConstraintDepth = 1000;

enum class ToolChoiceMode : std::uint8_t { Auto, None, Required };
// Basic protects every tool call. Automatic applies constraints only for strict tools or an
// explicit selection/count policy.
enum class ToolConstraintMode : std::uint8_t { Automatic, Basic };

// Engine-level tool selection (distinct from the wire-level serve::ToolChoice). Declarations
// belong to PromptOptions. Selection and cardinality affect this generation only.
// capy-flash default: Automatic (upstream: Basic), so that tool_choice:auto without strict tools
// keeps today's unconstrained path and its stop strings.
struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
    std::optional<std::vector<std::string>> allowed_names;
    bool parallel                  = true;
    ToolConstraintMode constraints = ToolConstraintMode::Automatic;
};

enum class ConstraintCacheAccess : std::uint8_t { Hit, Built, Waited };
enum class ConstraintOutputBranch : std::uint8_t { Undecided, Content, Tools };

// State describes the committed output language, including an assistant continuation prefix.
// Work includes speculative lookahead that was subsequently rolled back. Times are subintervals
// of existing request timings, not extra latency to add to them.
struct ConstraintObservation {
    ConstraintOutputBranch branch   = ConstraintOutputBranch::Undecided;
    bool complete                   = false;
    bool terminated                 = false;
    ConstraintCacheAccess cache     = ConstraintCacheAccess::Hit;
    bool timings_collected          = false;
    double prepare_seconds          = 0.0;
    double mask_seconds             = 0.0;
    double matcher_seconds          = 0.0;
    std::uint64_t mask_positions    = 0;
    std::uint64_t mask_upload_bytes = 0;
};

} // namespace ninfer
