#pragma once
#include "ninfer/types.h"
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::text {
// Reject unsupported constraints instead of letting a compiler silently weaken a schema.
void validate_structured_output(const StructuredOutputOptions& options);

class GrammarState {
public:
    ~GrammarState();
    GrammarState(GrammarState&&) noexcept;
    GrammarState& operator=(GrammarState&&) noexcept;
    [[nodiscard]] std::unique_ptr<GrammarState> fork() const;
    void accept(std::span<const TokenId> tokens);
    // Column i is conditioned on drafts[0..i). Unreachable suffix columns are unrestricted.
    void fill_masks(std::span<std::uint32_t> masks, std::span<const TokenId> drafts) const;
private:
    struct Impl;
    explicit GrammarState(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class StructuredCompiler;
};

class StructuredCompiler {
public:
    StructuredCompiler(std::vector<std::string> decoded_vocab, std::vector<int> stop_tokens);
    ~StructuredCompiler();
    std::shared_ptr<GrammarState> compile(const StructuredOutputOptions& options);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer::text
