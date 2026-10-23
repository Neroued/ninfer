#pragma once

#include "models/common/frontend/resources.h"

#include <mutex>

namespace ninfer::text {
class GrammarCompiler;
}

namespace ninfer::models::frontend {

// Immutable token resources with a shared, bounded compilation cache. Matchers belong to requests.
class OutputResources {
public:
    OutputResources(const TextResources& resources, std::size_t grammar_cache_bytes);
    ~OutputResources();
    text::GrammarCompiler& grammars() const;
    StopPolicy stop_policy(const StopPolicy& caller) const;
    const std::shared_ptr<const text::Tokenizer> tokenizer;
    const StopPolicy defaults;
private:
    const std::size_t cache_bytes_;
    mutable std::once_flag grammar_once_;
    mutable std::unique_ptr<text::GrammarCompiler> grammar_;
};

} // namespace ninfer::models::frontend
