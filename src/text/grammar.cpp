#include "text/grammar.h"
#include "text/json_schema.h"
#include "text/unicode.h"

#include <xgrammar/xgrammar.h>
#include "grammar_functor.h"
#include "grammar_impl.h"
#include "regex_converter.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace ninfer::text {
namespace {
class WorkTimer {
public:
    WorkTimer(bool enabled, double& seconds) : seconds_(enabled ? &seconds : nullptr) {
        if (seconds_) started_ = std::chrono::steady_clock::now();
    }

    ~WorkTimer() {
        if (seconds_)
            *seconds_ +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count();
    }
private:
    double* seconds_;
    std::chrono::steady_clock::time_point started_;
};

ConstraintCacheAccess cache_access(xgrammar::CompilationCacheAccess access) {
    switch (access) {
    case xgrammar::CompilationCacheAccess::kHit:
        return ConstraintCacheAccess::Hit;
    case xgrammar::CompilationCacheAccess::kBuilt:
        return ConstraintCacheAccess::Built;
    case xgrammar::CompilationCacheAccess::kWaited:
        return ConstraintCacheAccess::Waited;
    }
    throw std::logic_error("invalid compilation cache observation");
}

void validate_utf8(std::string_view value) {
    for (std::size_t offset = 0; offset < value.size();) {
        offset += unicode_internal::utf8_codepoint_at(value, offset, "output constraint").length;
    }
}

// Text publication consumes Unicode scalar values, including in negated regex classes.
class ScalarRegex final : public xgrammar::GrammarMutator {
    int32_t VisitCharacterClass(const GrammarExpr& expression) final {
        using Range = xgrammar::GrammarBuilder::CharacterClassElement;
        std::vector<Range> ranges;
        const bool negative = expression[0] != 0;
        for (int i = 1; i < expression.size(); i += 2) {
            const int lo = expression[i], hi = expression[i + 1];
            if (negative) {
                ranges.push_back({lo, hi});
            } else {
                if (lo <= 0xd7ff) ranges.push_back({lo, std::min(hi, 0xd7ff)});
                if (hi >= 0xe000) ranges.push_back({std::max(lo, 0xe000), hi});
            }
        }
        if (negative) ranges.push_back({0xd800, 0xdfff});
        std::sort(ranges.begin(), ranges.end(),
                  [](const auto& a, const auto& b) { return a.lower < b.lower; });
        std::vector<Range> merged;
        for (const auto& range : ranges) {
            if (!merged.empty() && range.lower <= merged.back().upper + 1)
                merged.back().upper = std::max(merged.back().upper, range.upper);
            else
                merged.push_back(range);
        }
        return builder_->AddCharacterClass(merged, negative);
    }

    int32_t VisitCharacterClassStar(const GrammarExpr& expression) final {
        return builder_->AddRepeatFromExpr("scalar", VisitCharacterClass(expression), 0, -1);
    }
};

xgrammar::Grammar choice_grammar(const std::vector<std::string>& choices) {
    if (choices.empty()) throw std::invalid_argument("choice requires at least one string");
    xgrammar::GrammarBuilder builder;
    std::vector<int32_t> alternatives;
    alternatives.reserve(choices.size());
    for (const auto& value : choices) {
        validate_utf8(value);
        alternatives.push_back(builder.AddByteString(value));
    }
    return xgrammar::GrammarNormalizer::Apply(
        builder.Get(builder.AddRuleWithHint("root", builder.AddChoices(alternatives))));
}

std::string quoted(unsigned char c) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result   = "\"\\x00\"";
    result[3]            = hex[c >> 4];
    result[4]            = hex[c & 15];
    return result;
}

// A small DFA consumes through the first complete delimiter, including delimiters split across
// tokens. Concatenation keeps the user's rule names in their own namespace.
xgrammar::Grammar reasoning_prefix(std::string_view delimiter) {
    std::string alphabet(delimiter);
    std::sort(alphabet.begin(), alphabet.end());
    alphabet.erase(std::unique(alphabet.begin(), alphabet.end()), alphabet.end());
    std::string source = "root ::= s0\n";
    for (std::size_t state = 0; state < delimiter.size(); ++state) {
        source += "s" + std::to_string(state) + " ::= ";
        for (unsigned char c : alphabet) {
            std::string next(delimiter.substr(0, state));
            next += static_cast<char>(c);
            std::size_t matched = std::min(next.size(), delimiter.size());
            while (matched && !next.ends_with(delimiter.substr(0, matched))) { --matched; }
            source += quoted(c);
            if (matched != delimiter.size()) { source += " s" + std::to_string(matched); }
            source += " | ";
        }
        source += "[^";
        for (unsigned char c : alphabet) {
            const auto escaped = quoted(c);
            source += escaped.substr(1, escaped.size() - 2);
        }
        source += "] s0\n";
    }
    return xgrammar::Grammar::FromEBNF(source);
}

void validate(const xgrammar::Grammar& grammar, const std::vector<std::string>& vocab,
              const std::vector<std::int32_t>& eos) {
    for (int i = 0; i < grammar->NumRules(); ++i) {
        const auto& rule = grammar->GetRule(i);
        if (rule.max_tokens >= 0 || rule.max_chars >= 0 || !rule.capture_name.empty() ||
            rule.is_lazy || rule.temperature || grammar->GetSuffixStopInfo(i)) {
            throw std::invalid_argument(
                "GBNF rules cannot set budgets, captures, lazy matching or temperature");
        }
    }
    using Type = xgrammar::Grammar::Impl::GrammarExprType;
    for (int i = 0; i < grammar->NumGrammarExprs(); ++i) {
        const auto expression = grammar->GetGrammarExpr(i);
        if (expression.type == Type::kTagDispatch || expression.type == Type::kTokenTagDispatch ||
            expression.type == Type::kRegex || expression.type == Type::kSubstring) {
            throw std::invalid_argument("grammar uses an extension outside the GBNF contract");
        }
        if (expression.type == Type::kToken || expression.type == Type::kExcludeToken) {
            for (int id : expression) {
                if (id < 0 || static_cast<std::size_t>(id) >= vocab.size() || vocab[id].empty() ||
                    std::find(eos.begin(), eos.end(), id) != eos.end()) {
                    throw std::invalid_argument(
                        "GBNF token references must name ordinary vocabulary tokens");
                }
            }
        }
    }
}
} // namespace

RequestErrorKind constraint_error_kind(OutputConstraintKind kind) {
    switch (kind) {
    case OutputConstraintKind::Grammar:
        return RequestErrorKind::InvalidGrammar;
    case OutputConstraintKind::Choice:
        return RequestErrorKind::InvalidChoice;
    case OutputConstraintKind::Regex:
        return RequestErrorKind::InvalidRegex;
    case OutputConstraintKind::JsonObject:
    case OutputConstraintKind::JsonSchema:
        return RequestErrorKind::InvalidJsonSchema;
    }
    throw std::invalid_argument("unknown output constraint kind");
}

class GrammarCompiler::Impl {
public:
    Impl(std::vector<std::string> vocab_, std::vector<std::int32_t> eos_, std::size_t bytes)
        : vocab(std::move(vocab_)), eos(std::move(eos_)),
          tokenizer(vocab, xgrammar::VocabType::RAW, static_cast<int>(vocab.size()), eos),
          compiler(tokenizer, 1, true, static_cast<std::int64_t>(bytes)) {}

    std::vector<std::string> vocab;
    std::vector<std::int32_t> eos;
    xgrammar::TokenizerInfo tokenizer;
    xgrammar::GrammarCompiler compiler;
};

class GrammarSession::Impl {
public:
    Impl(const xgrammar::CompiledGrammar& grammar, int vocab)
        : matcher(grammar), vocabulary(vocab), words((vocab + 31) / 32) {}

    xgrammar::GrammarMatcher matcher;
    int vocabulary;
    std::size_t words;
    int tentative = 0;
    ConstraintObservation observed;

    void fill(std::span<std::uint32_t> target) {
        if (matcher.IsTerminated()) {
            std::fill(target.begin(), target.end(), 0);
            return;
        }
        std::int64_t shape[] = {1, static_cast<std::int64_t>(words)};
        DLTensor tensor{target.data(), {kDLCPU, 0}, 2, {kDLInt, 32, 1}, shape, nullptr, 0};
        matcher.FillNextTokenBitmask(&tensor);
        if (vocabulary % 32) { target.back() &= (1u << (vocabulary % 32)) - 1u; }
    }
};

GrammarCompiler::GrammarCompiler(std::vector<std::string> vocab, std::vector<std::int32_t> eos,
                                 std::size_t bytes)
    : impl_(std::make_unique<Impl>(std::move(vocab), std::move(eos), bytes)) {}

GrammarCompiler::~GrammarCompiler() = default;

std::unique_ptr<GrammarSession>
GrammarCompiler::compile_model(std::string_view identity,
                               const std::function<xgrammar::Grammar()>& build,
                               std::string_view close, std::string_view continuation) {
    const std::string key =
        "model:" + std::to_string(close.size()) + ":" + std::string(close) + std::string(identity);
    xgrammar::CompilationCacheAccess access;
    auto compiled = impl_->compiler.CompileCachedGrammar(
        key,
        [&] {
            auto grammar = build();
            return close.empty() ? grammar
                                 : xgrammar::Grammar::Concat({reasoning_prefix(close), grammar});
        },
        &access);
    auto session =
        std::make_unique<GrammarSession::Impl>(compiled, static_cast<int>(impl_->vocab.size()));
    session->observed.cache = cache_access(access);
    if (!continuation.empty() && !session->matcher.AcceptString(std::string(continuation)))
        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                           "assistant continuation is not a prefix of the tool grammar");
    return std::unique_ptr<GrammarSession>(new GrammarSession(std::move(session)));
}

std::unique_ptr<GrammarSession> GrammarCompiler::compile(const OutputConstraint& constraint,
                                                         std::string_view close,
                                                         std::string_view continuation) {
    const auto error_kind = constraint_error_kind(constraint.kind);
    try {
        auto choices = constraint.choices;
        if (constraint.kind == OutputConstraintKind::Choice) {
            if (!constraint.source.empty())
                throw std::invalid_argument("choice uses literal alternatives, not source text");
            std::sort(choices.begin(), choices.end());
            choices.erase(std::unique(choices.begin(), choices.end()), choices.end());
        } else if (!choices.empty()) {
            throw std::invalid_argument("literal alternatives require a choice constraint");
        }
        std::string key = std::to_string(static_cast<int>(constraint.kind)) + ":" +
                          std::to_string(close.size()) + ":" + std::string(close) +
                          constraint.source;
        for (const auto& value : choices) key += std::to_string(value.size()) + ":" + value;
        xgrammar::CompilationCacheAccess access;
        auto compiled = impl_->compiler.CompileCachedGrammar(
            key,
            [&] {
                auto grammar = [&]() -> xgrammar::Grammar {
                    if (constraint.kind == OutputConstraintKind::Grammar) {
                        auto parsed = xgrammar::Grammar::FromEBNF(constraint.source, "root");
                        validate(parsed, impl_->vocab, impl_->eos);
                        return parsed;
                    } else if (constraint.kind == OutputConstraintKind::Choice) {
                        return choice_grammar(choices);
                    } else if (constraint.kind == OutputConstraintKind::Regex) {
                        validate_utf8(constraint.source);
                        return xgrammar::GrammarNormalizer::Apply(
                            ScalarRegex().Apply(xgrammar::Grammar::FromRegex(
                                xgrammar::NormalizeRegexPattern(constraint.source, true))));
                    } else {
                        return build_json_grammar(constraint);
                    }
                }();
                if (!close.empty())
                    grammar = xgrammar::Grammar::Concat({reasoning_prefix(close), grammar});
                return grammar;
            },
            &access);
        auto session =
            std::make_unique<GrammarSession::Impl>(compiled, static_cast<int>(impl_->vocab.size()));
        session->observed.cache = cache_access(access);
        if (!continuation.empty() && !session->matcher.AcceptString(std::string(continuation))) {
            throw std::invalid_argument("assistant continuation is not a prefix of the grammar");
        }
        return std::unique_ptr<GrammarSession>(new GrammarSession(std::move(session)));
    } catch (const xgrammar::JSONSchemaCompileError& error) {
        RequestErrorKind kind = RequestErrorKind::InvalidJsonSchema;
        if (error.kind == xgrammar::SchemaErrorType::kUnsupportedSchema)
            kind = RequestErrorKind::UnsupportedJsonSchema;
        else if (error.kind == xgrammar::SchemaErrorType::kUnsatisfiableSchema)
            kind = RequestErrorKind::UnsatisfiableJsonSchema;
        throw RequestError(kind,
                           std::string(error.what()) + " at " +
                               (error.pointer.empty() ? "/" : error.pointer),
                           error.pointer);
    } catch (const xgrammar::LogFatalError& error) {
        throw RequestError(error_kind, error.what());
    } catch (const RequestError&) { throw; } catch (const std::invalid_argument& error) {
        throw RequestError(error_kind, error.what());
    }
}

GrammarSession::GrammarSession(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

GrammarSession::~GrammarSession()                                    = default;
GrammarSession::GrammarSession(GrammarSession&&) noexcept            = default;
GrammarSession& GrammarSession::operator=(GrammarSession&&) noexcept = default;

std::size_t GrammarSession::mask_words() const noexcept { return impl_->words; }

std::uint32_t GrammarSession::masks(std::span<const std::int32_t> drafts,
                                    std::span<std::uint32_t> words) {
    if (drafts.size() >= 32 || impl_->tentative ||
        words.size() != (drafts.size() + 1) * impl_->words) {
        throw std::logic_error("invalid grammar lookahead transaction");
    }
    WorkTimer timer(impl_->observed.timings_collected, impl_->observed.mask_seconds);
    int advanced       = 0;
    bool reachable     = true;
    std::uint32_t dead = 0;
    try {
        for (std::size_t position = 0; position <= drafts.size(); ++position) {
            auto mask = words.subspan(position * impl_->words, impl_->words);
            if (reachable && !impl_->matcher.IsTerminated()) {
                impl_->fill(mask);
                ++impl_->observed.mask_positions;
                if (std::all_of(mask.begin(), mask.end(), [](auto word) { return word == 0; })) {
                    dead |= 1u << position;
                    reachable = false;
                    mask[0]   = 1;
                }
            } else {
                std::fill(mask.begin(), mask.end(), 0);
                mask[0] = 1;
            }
            if (position == drafts.size()) { break; }
            // Rejected proposals are expected; advance only through the allowed token set.
            const auto draft = drafts[position];
            if (reachable && !impl_->matcher.IsTerminated() && draft >= 0 &&
                draft < impl_->vocabulary && (mask[draft / 32] & (1u << (draft % 32))) != 0 &&
                impl_->matcher.AcceptToken(draft)) {
                ++advanced;
            } else {
                reachable = false;
            }
        }
    } catch (...) {
        if (advanced) { impl_->matcher.Rollback(advanced); }
        throw;
    }
    if (advanced) { impl_->matcher.Rollback(advanced); }
    return dead;
}

void GrammarSession::accept(std::int32_t token) { accept(std::span(&token, 1)); }

void GrammarSession::accept(std::span<const std::int32_t> tokens) {
    WorkTimer timer(impl_->observed.timings_collected, impl_->observed.matcher_seconds);
    for (const auto token : tokens) {
        if (impl_->matcher.IsTerminated() || !impl_->matcher.AcceptToken(token))
            throw std::logic_error("selected token violates the request grammar");
        ++impl_->tentative;
    }
}

void GrammarSession::observe(bool timings, double prepare_seconds) noexcept {
    impl_->observed.timings_collected = timings;
    impl_->observed.prepare_seconds   = timings ? prepare_seconds : 0.0;
}

void GrammarSession::uploaded(std::size_t bytes) noexcept {
    impl_->observed.mask_upload_bytes += bytes;
}

ConstraintObservation GrammarSession::observation() const {
    if (impl_->tentative) throw std::logic_error("cannot publish tentative grammar state");
    auto result       = impl_->observed;
    result.terminated = impl_->matcher.IsTerminated();
    result.complete   = result.terminated || impl_->matcher.IsCompleted();
    return result;
}

void GrammarSession::confirm() noexcept { impl_->tentative = 0; }

void GrammarSession::discard() {
    WorkTimer timer(impl_->observed.timings_collected, impl_->observed.matcher_seconds);
    if (impl_->tentative) { impl_->matcher.Rollback(impl_->tentative); }
    impl_->tentative = 0;
}
} // namespace ninfer::text
