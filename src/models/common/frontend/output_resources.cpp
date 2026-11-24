#include "models/common/frontend/output_resources.h"
#include "text/grammar.h"
#include "text/unicode.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::frontend {
namespace {
StopPolicy merge_stop_policy(const text::Tokenizer& tokenizer, std::span<const TokenId> defaults,
                             const StopPolicy& caller) {
    StopPolicy result;
    result.publish_stop_token = caller.publish_stop_token;
    const auto append_token   = [&](TokenId token) {
        if (!tokenizer.is_valid_token(token)) {
            throw std::invalid_argument("stop token id is outside the checkpoint vocabulary: " +
                                          std::to_string(token));
        }
        if (std::find(result.token_ids.begin(), result.token_ids.end(), token) ==
            result.token_ids.end()) {
            result.token_ids.push_back(token);
        }
    };
    if (caller.include_model_defaults) {
        for (const int token : defaults) { append_token(token); }
    }
    for (const TokenId token : caller.token_ids) { append_token(token); }

    result.strings.reserve(caller.strings.size());
    for (const StopString& stop : caller.strings) {
        if (stop.text.empty()) { throw std::invalid_argument("stop string must not be empty"); }
        (void)text::unicode_internal::utf8_codepoints(stop.text, "stop string");
        const auto duplicate = std::find_if(
            result.strings.begin(), result.strings.end(), [&](const StopString& existing) {
                return existing.text == stop.text && existing.channel == stop.channel &&
                       existing.include_in_output == stop.include_in_output;
            });
        if (duplicate == result.strings.end()) { result.strings.push_back(stop); }
    }
    return result;
}

} // namespace

OutputResources::OutputResources(const TextResources& resources, std::size_t cache_bytes)
    : tokenizer(resources.tokenizer), defaults{.token_ids = resources.stop_token_ids},
      cache_bytes_(cache_bytes) {
    if (!tokenizer) throw std::invalid_argument("output resources require a tokenizer");
}

OutputResources::~OutputResources() = default;

StopPolicy OutputResources::stop_policy(const StopPolicy& caller) const {
    return merge_stop_policy(*tokenizer, defaults.token_ids, caller);
}

text::GrammarCompiler& OutputResources::grammars() const {
    std::call_once(grammar_once_, [&] {
        std::vector<std::string> vocab(tokenizer->vocab_size());
        for (std::size_t id = 0; id < vocab.size(); ++id) {
            if (!tokenizer->is_valid_token(static_cast<TokenId>(id))) continue;
            const auto decoded = tokenizer->decoded_token(static_cast<TokenId>(id));
            if (!decoded.special) vocab[id] = decoded.bytes;
        }
        grammar_ = std::make_unique<text::GrammarCompiler>(std::move(vocab), defaults.token_ids,
                                                           cache_bytes_);
    });
    return *grammar_;
}

} // namespace ninfer::models::frontend
