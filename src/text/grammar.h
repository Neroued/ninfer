#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace xgrammar {
class Grammar;
}

namespace ninfer::text {

[[nodiscard]] RequestErrorKind constraint_error_kind(OutputConstraintKind kind);

// One immutable vocabulary/compiler per model; one transactional matcher per request.
class GrammarSession {
public:
    ~GrammarSession();
    GrammarSession(GrammarSession&&) noexcept;
    GrammarSession& operator=(GrammarSession&&) noexcept;
    [[nodiscard]] std::uint32_t masks(std::span<const std::int32_t> drafts,
                                      std::span<std::uint32_t> words);
    void accept(std::int32_t token);
    void accept(std::span<const std::int32_t> tokens);
    void confirm() noexcept;
    void discard();
    [[nodiscard]] std::size_t mask_words() const noexcept;
    void observe(bool timings, double prepare_seconds = 0.0) noexcept;
    void uploaded(std::size_t bytes) noexcept;
    [[nodiscard]] ConstraintObservation observation() const;

private:
    class Impl;
    explicit GrammarSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class GrammarCompiler;
};

class GrammarCompiler {
public:
    // Empty entries explicitly identify forbidden/special IDs. EOS is supplied separately.
    GrammarCompiler(std::vector<std::string> vocabulary, std::vector<std::int32_t> eos,
                    std::size_t cache_bytes);
    ~GrammarCompiler();
    [[nodiscard]] std::unique_ptr<GrammarSession> compile(const OutputConstraint& constraint,
                                                          std::string_view reasoning_close,
                                                          std::string_view continuation);
    // Model-owned composition, built only on a miss in the same bounded compiler cache.
    [[nodiscard]] std::unique_ptr<GrammarSession>
    compile_model(std::string_view identity, const std::function<xgrammar::Grammar()>& build,
                  std::string_view reasoning_close, std::string_view continuation);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::text
