#include "text/grammar.h"
#include "text/constraint_limits.h"
#include "text/json_schema.h"
#include "text/unicode.h"

#include <xgrammar/xgrammar.h>
#include "fsm_builder.h"
#include "grammar_functor.h"
#include "grammar_impl.h"
#include "json_string_grammar.h"
#include "regex_converter.h"
#include "support/work_budget.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

#if defined(__GLIBC__)
#    include <malloc.h>
#endif

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

// Reasoning frame: any bytes through the FIRST complete delimiter, as the deterministic
// string-matching (KMP) automaton of the delimiter whose only end state has no outgoing edge.
// - First occurrence: the frame ends exactly where OutputSession closes the reasoning channel, so
//   a second close after the canonical one is content, which the constrained language rejects
//   (before, "…\n</think>\n" + "\n</think>\n\n" parsed as body + delimiter and a duplicate close
//   reached the published content).
// - Bytes, not Unicode scalars: a thinking budget can stop the model between the byte-level tokens
//   of one character; the forced control suffix must remain in the language there, as in free
//   generation (the decoder publishes U+FFFD for the cut character).
xgrammar::Grammar reasoning_prefix(std::string_view delimiter) {
    if (delimiter.empty()) throw std::logic_error("invalid reasoning delimiter");
    const std::size_t size = delimiter.size();
    std::vector<std::size_t> failure(size, 0);
    for (std::size_t index = 1, matched = 0; index < size; ++index) {
        while (matched != 0 && delimiter[index] != delimiter[matched])
            matched = failure[matched - 1];
        if (delimiter[index] == delimiter[matched]) ++matched;
        failure[index] = matched;
    }
    xgrammar::FSM fsm(static_cast<int>(size + 1));
    for (std::size_t state = 0; state < size; ++state) {
        std::array<int, 256> next{};
        for (int byte = 0; byte < 256; ++byte) {
            std::size_t matched = state;
            while (matched != 0 && static_cast<unsigned char>(delimiter[matched]) != byte)
                matched = failure[matched - 1];
            if (static_cast<unsigned char>(delimiter[matched]) == byte) ++matched;
            next[byte] = static_cast<int>(matched);
        }
        for (int low = 0; low < 256;) {
            int high = low;
            while (high + 1 < 256 && next[high + 1] == next[low]) ++high;
            fsm.AddEdge(static_cast<int>(state), next[low], low, high);
            low = high + 1;
        }
    }
    xgrammar::GrammarBuilder builder;
    const auto value = builder.AddRuleRef(builder.AddRuleWithHint(
        "reasoning",
        builder.AddByteFSM(xgrammar::FSMWithStartEnd(fsm, 0, {static_cast<std::int32_t>(size)}))));
    return xgrammar::GrammarNormalizer::Apply(builder.Get(builder.AddRuleWithHint("root", value)));
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
            expression.type == Type::kRegex || expression.type == Type::kByteFSM ||
            expression.type == Type::kSubstring) {
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

using Clock = std::chrono::steady_clock;

std::string milliseconds(std::uint32_t value) { return std::to_string(value) + " ms"; }

// capy-flash: at most `limit` compilations at once (0 = no limit); cache hits never take a slot.
class CompileSlots {
public:
    explicit CompileSlots(std::uint32_t limit) : limit_(limit) {}

    class Slot {
    public:
        explicit Slot(CompileSlots& owner) : owner_(&owner) {}

        Slot(const Slot&)            = delete;
        Slot& operator=(const Slot&) = delete;

        ~Slot() {
            {
                std::lock_guard lock(owner_->mutex_);
                --owner_->busy_;
            }
            owner_->freed_.notify_all();
        }

    private:
        CompileSlots* owner_;
    };

    // False when no slot was free before the deadline. First come, first served: a waiter is
    // admitted only when every earlier waiter has been, so a stream of new compilations cannot
    // starve an earlier one.
    bool acquire(std::optional<Slot>& slot, std::optional<Clock::time_point> deadline) {
        if (limit_ == 0) return true;
        std::unique_lock lock(mutex_);
        const std::uint64_t ticket = next_ticket_++;
        const auto turn            = [&] {
            return busy_ < limit_ && (waiting_.empty() || waiting_.front() == ticket);
        };
        waiting_.push_back(ticket);
        const bool admitted =
            deadline ? freed_.wait_until(lock, *deadline, turn) : (freed_.wait(lock, turn), true);
        waiting_.erase(std::find(waiting_.begin(), waiting_.end(), ticket));
        if (!admitted) {
            lock.unlock();
            freed_.notify_all(); // the next waiter may now be at the front
            return false;
        }
        ++busy_;
        slot.emplace(*this);
        lock.unlock();
        freed_.notify_all();
        return true;
    }

private:
    const std::uint32_t limit_;
    std::mutex mutex_;
    std::condition_variable freed_;
    std::uint32_t busy_        = 0;
    std::uint64_t next_ticket_ = 0;
    std::deque<std::uint64_t> waiting_;
};

// Memory of a large or failed compilation goes back to the system instead of staying in the
// allocator arenas of the HTTP thread (RSS grew by GB and never came down).
void release_compilation_memory() {
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

// M4: a language without any first token (an empty character class, [^\s\S]...) is rejected
// here instead of failing after the prefill. Only the content language is probed: with thinking
// the probe first crosses the canonical reasoning close.
bool accepts_no_output(const xgrammar::CompiledGrammar& compiled, std::string_view close,
                       std::size_t words) {
    xgrammar::GrammarMatcher probe(compiled);
    if (!close.empty() && !probe.AcceptString(std::string(close))) return false;
    std::vector<std::uint32_t> mask(words, 0U);
    std::int64_t shape[] = {1, static_cast<std::int64_t>(words)};
    DLTensor tensor{mask.data(), {kDLCPU, 0}, 2, {kDLInt, 32, 1}, shape, nullptr, 0};
    probe.FillNextTokenBitmask(&tensor);
    return std::all_of(mask.begin(), mask.end(), [](std::uint32_t word) { return word == 0; });
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

class GrammarVocabulary::Impl {
public:
    explicit Impl(std::vector<std::string> value) : pieces(std::move(value)) {}

    std::vector<std::string> pieces;
    mutable std::mutex mutex;
    mutable std::map<unsigned char, std::vector<std::int32_t>> indexes;
};

GrammarVocabulary::GrammarVocabulary(std::vector<std::string> pieces)
    : impl_(std::make_unique<Impl>(std::move(pieces))) {}

GrammarVocabulary::~GrammarVocabulary() = default;

const std::vector<std::string>& GrammarVocabulary::pieces() const noexcept { return impl_->pieces; }

const std::vector<std::int32_t>& GrammarVocabulary::containing(unsigned char byte) const {
    std::lock_guard lock(impl_->mutex);
    const auto found = impl_->indexes.find(byte);
    if (found != impl_->indexes.end()) return found->second;
    std::vector<std::int32_t> ids;
    for (std::size_t i = 0; i < impl_->pieces.size(); ++i)
        if (impl_->pieces[i].find(static_cast<char>(byte)) != std::string::npos)
            ids.push_back(static_cast<std::int32_t>(i));
    return impl_->indexes.emplace(byte, std::move(ids)).first->second;
}

class GrammarCompiler::Impl {
public:
    Impl(std::vector<std::string> vocab_, std::vector<std::int32_t> eos_, std::size_t bytes,
         const OutputConstraintLimits& limits_)
        : vocabulary(std::make_shared<GrammarVocabulary>(std::move(vocab_))),
          vocab(vocabulary->pieces()), eos(std::move(eos_)),
          tokenizer(vocab, xgrammar::VocabType::RAW, static_cast<int>(vocab.size()), eos),
          compiler(tokenizer, 1, true, static_cast<std::int64_t>(bytes)), limits(limits_),
          slots(limits_.compile_concurrency) {}

    std::shared_ptr<const GrammarVocabulary> vocabulary;
    const std::vector<std::string>& vocab;
    std::vector<std::int32_t> eos;
    xgrammar::TokenizerInfo tokenizer;
    xgrammar::GrammarCompiler compiler;
    OutputConstraintLimits limits;
    CompileSlots slots;
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
    std::shared_ptr<const GrammarVocabulary> token_bytes;
    std::unique_ptr<GrammarStateConstraint> state;

    // capy-flash matcher budget (OutputConstraintLimits): per call, cumulative as
    // request + allowance x rounds, and history memory. Once `failure` is set every mask is dead,
    // so the Engine closes the request at its next round. After an interrupted parser step
    // (`interrupted`) the matcher is in an unspecified state and is never touched again; otherwise
    // it stays consistent, so a pending accept (a control span) still succeeds.
    OutputConstraintLimits limits;
    std::chrono::nanoseconds spent{}; // matcher CPU time of this request (second audit N3)
    std::uint64_t rounds = 0;
    std::optional<std::string> failure;
    bool interrupted = false;

    // The budgets count the CPU time of the calling thread, not the wall clock: on a busy machine
    // (Flash-Next's expert threads, concurrent compilations, other processes) a legitimate grammar
    // must not look expensive. The wall clock stays as a wider safety net (5x the round budget),
    // so a step that stalls the batch is still interrupted.
    struct CallStart {
        Clock::time_point wall       = Clock::now();
        std::chrono::nanoseconds cpu = xgrammar::ThreadCpuTime();
    };

    void install(const CallStart& start, std::optional<xgrammar::ScopedWorkDeadline>& work) const {
        if (limits.round_budget_ms == 0) return;
        const auto round = std::chrono::milliseconds(limits.round_budget_ms);
        work.emplace(start.wall + 5 * round, start.cpu + round);
    }

    static std::chrono::nanoseconds cpu_since(const CallStart& start) {
        return xgrammar::ThreadCpuTime() - start.cpu;
    }

    void fail_round(bool stopped = false) {
        interrupted = interrupted || stopped;
        failure     = "output constraint exceeded its matcher budget of " +
                  milliseconds(limits.round_budget_ms) +
                  " of CPU time per decode round (--constraint-round-budget-ms); the grammar is "
                  "too ambiguous or expensive for constrained decoding";
    }

    // Adds one call's time to the cumulative budget; true when the request is over it. The
    // per-round limit is enforced inside the parser (WorkDeadline) on the parser's own work; a
    // rare long step outside it (the history buffer doubling on a long output) only counts here.
    bool charge(std::chrono::nanoseconds elapsed, bool round) {
        spent += elapsed;
        if (round) ++rounds;
        if (limits.request_budget_ms == 0) return false;
        const auto allowed = std::chrono::milliseconds(limits.request_budget_ms) +
                             std::chrono::milliseconds(limits.round_allowance_ms) * rounds;
        if (spent <= allowed) return false;
        const auto used = std::chrono::duration_cast<std::chrono::milliseconds>(spent).count();
        failure = "output constraint used " + std::to_string(used) + " ms of matcher CPU time in " +
                  std::to_string(rounds) + " decode rounds; the budget is " +
                  milliseconds(limits.request_budget_ms) + " plus " +
                  milliseconds(limits.round_allowance_ms) +
                  " per round (--constraint-request-budget-ms, --constraint-round-allowance-ms)";
        return true;
    }

    // Every position dead with a one-token safe mask, as for a real dead end.
    std::uint32_t all_dead(std::span<std::uint32_t> target, std::size_t positions) const {
        std::fill(target.begin(), target.end(), 0U);
        for (std::size_t position = 0; position < positions; ++position)
            target[position * words] = 1U;
        return positions >= 32 ? ~0U : (1U << positions) - 1U;
    }

    void fill(std::span<std::uint32_t> target) {
        if (matcher.IsTerminated()) {
            std::fill(target.begin(), target.end(), 0);
            return;
        }
        std::int64_t shape[] = {1, static_cast<std::int64_t>(words)};
        DLTensor tensor{target.data(), {kDLCPU, 0}, 2, {kDLInt, 32, 1}, shape, nullptr, 0};
        matcher.FillNextTokenBitmask(&tensor);
        if (vocabulary % 32) { target.back() &= (1u << (vocabulary % 32)) - 1u; }
        if (state) state->filter(target);
    }
};

GrammarCompiler::GrammarCompiler(std::vector<std::string> vocab, std::vector<std::int32_t> eos,
                                 std::size_t bytes, const OutputConstraintLimits& limits)
    : impl_(std::make_unique<Impl>(std::move(vocab), std::move(eos), bytes, limits)) {}

GrammarCompiler::~GrammarCompiler() = default;

std::shared_ptr<const GrammarVocabulary> GrammarCompiler::vocabulary() const {
    return impl_->vocabulary;
}

std::unique_ptr<GrammarSession>
GrammarCompiler::compile_model(std::string_view identity,
                               const std::function<xgrammar::Grammar()>& build,
                               std::string_view close, std::string_view continuation,
                               std::unique_ptr<GrammarStateConstraint> state) {
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
    session->token_bytes    = impl_->vocabulary;
    session->state          = std::move(state);
    session->limits         = impl_->limits;
    if (!continuation.empty()) {
        if (!session->matcher.AcceptString(std::string(continuation)) ||
            (session->state && !session->state->accept(continuation)))
            throw RequestError(RequestErrorKind::InvalidToolConstraint,
                               "assistant continuation is not a prefix of the tool constraint");
        if (session->state) session->state->confirm();
    }
    return std::unique_ptr<GrammarSession>(new GrammarSession(std::move(session)));
}

std::unique_ptr<GrammarSession> GrammarCompiler::compile(const OutputConstraint& constraint,
                                                         std::string_view close,
                                                         std::string_view continuation) {
    const auto error_kind = constraint_error_kind(constraint.kind);
    const auto& limits    = impl_->limits;
    const auto started    = Clock::now();
    // capy-flash: the compilation itself runs under --constraint-compile-timeout-ms, counted from
    // the moment it holds a slot; the wait for a slot is bounded apart by
    // --constraint-compile-wait-ms and ends in "busy" (429), so slow schemas of one client cannot
    // make the legitimate schemas of others look too complex.
    const std::optional<std::chrono::milliseconds> timeout =
        limits.compile_timeout_ms != 0
            ? std::optional(std::chrono::milliseconds(limits.compile_timeout_ms))
            : std::nullopt;
    const std::optional<Clock::time_point> deadline =
        timeout ? std::optional(started + *timeout) : std::nullopt;
    const std::optional<Clock::time_point> wait_deadline =
        limits.compile_wait_ms != 0
            ? std::optional(started + std::chrono::milliseconds(limits.compile_wait_ms))
            : std::nullopt;
    std::optional<CompileSlots::Slot> slot;
    bool built = false;
    try {
        check_constraint_limits(constraint, limits);
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
        std::optional<xgrammar::ScopedWorkDeadline> work;
        if (deadline) work.emplace(*deadline);
        xgrammar::CompilationCacheAccess access;
        auto compiled = impl_->compiler.CompileCachedGrammar(
            key,
            [&] {
                // Cache miss: the compilation (mask cache included) holds a slot until the end.
                built = true;
                if (!impl_->slots.acquire(slot, wait_deadline)) {
                    throw RequestError(
                        RequestErrorKind::Overloaded,
                        "grammar compiler busy: no compilation slot became free within " +
                            milliseconds(limits.compile_wait_ms) +
                            " (--constraint-compile-wait-ms, --constraint-compile-concurrency); "
                            "retry later");
                }
                if (timeout) work.emplace(Clock::now() + *timeout);
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
                        return build_json_grammar(constraint, limits);
                    }
                }();
                if (work) work->Check();
                if (!close.empty())
                    grammar = xgrammar::Grammar::Concat({reasoning_prefix(close), grammar});
                return grammar;
            },
            &access);
        auto session =
            std::make_unique<GrammarSession::Impl>(compiled, static_cast<int>(impl_->vocab.size()));
        session->observed.cache = cache_access(access);
        session->limits         = limits;
        if (accepts_no_output(compiled, close, session->words)) {
            throw RequestError(constraint.kind == OutputConstraintKind::JsonSchema
                                   ? RequestErrorKind::UnsatisfiableJsonSchema
                                   : error_kind,
                               "the output constraint accepts no output (its language is empty)");
        }
        if (!continuation.empty() && !session->matcher.AcceptString(std::string(continuation))) {
            throw std::invalid_argument("assistant continuation is not a prefix of the grammar");
        }
        work.reset();
        slot.reset();
        if (built && Clock::now() - started > std::chrono::milliseconds(100))
            release_compilation_memory();
        return std::unique_ptr<GrammarSession>(new GrammarSession(std::move(session)));
    } catch (const xgrammar::WorkBudgetExceeded&) {
        slot.reset();
        release_compilation_memory();
        throw RequestError(RequestErrorKind::ConstraintLimitExceeded,
                           "output constraint compilation exceeded " +
                               milliseconds(limits.compile_timeout_ms) +
                               " (--constraint-compile-timeout-ms); simplify the constraint");
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
    const std::size_t positions = drafts.size() + 1;
    if (impl_->failure) return impl_->all_dead(words, positions);
    const Impl::CallStart started;
    std::optional<xgrammar::ScopedWorkDeadline> work;
    impl_->install(started, work);
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
                if (impl_->state && !impl_->state->accept(impl_->token_bytes->pieces()[draft]))
                    throw std::logic_error("grammar state mask admitted an invalid draft");
            } else {
                reachable = false;
            }
        }
    } catch (const xgrammar::WorkBudgetExceeded&) {
        // The interrupted parser is not rolled back: the session is failed for good.
        if (impl_->state) impl_->state->discard();
        impl_->fail_round(true);
        return impl_->all_dead(words, positions);
    } catch (...) {
        if (advanced) { impl_->matcher.Rollback(advanced); }
        if (impl_->state) impl_->state->discard();
        throw;
    }
    if (advanced) { impl_->matcher.Rollback(advanced); }
    if (impl_->state) impl_->state->discard();
    work.reset();
    if (impl_->charge(Impl::cpu_since(started), true)) return impl_->all_dead(words, positions);
    return dead;
}

void GrammarSession::accept(std::int32_t token) { accept(std::span(&token, 1)); }

void GrammarSession::accept(std::span<const std::int32_t> tokens) {
    WorkTimer timer(impl_->observed.timings_collected, impl_->observed.matcher_seconds);
    if (impl_->interrupted) throw std::logic_error(*impl_->failure);
    const Impl::CallStart started;
    std::optional<xgrammar::ScopedWorkDeadline> work;
    impl_->install(started, work);
    try {
        for (const auto token : tokens) {
            if (impl_->matcher.IsTerminated() || !impl_->matcher.AcceptToken(token))
                throw std::logic_error("selected token violates the request grammar");
            ++impl_->tentative;
            if (impl_->state && !impl_->state->accept(impl_->token_bytes->pieces()[token])) {
                discard();
                throw std::logic_error("selected token violates the request grammar state");
            }
        }
    } catch (const xgrammar::WorkBudgetExceeded&) {
        impl_->fail_round(true);
        throw std::logic_error(*impl_->failure);
    }
    work.reset();
    // A cumulative overrun is reported by the next masks(): these tokens were legal.
    if (!impl_->failure && !impl_->charge(Impl::cpu_since(started), false) &&
        impl_->limits.matcher_bytes != 0 &&
        impl_->matcher.GetHistoryMemoryBytes() > impl_->limits.matcher_bytes) {
        impl_->failure = "output constraint matcher history exceeded " +
                         std::to_string(impl_->limits.matcher_bytes >> 20) +
                         " MiB (--constraint-matcher-mib); the grammar is too ambiguous for this "
                         "output length";
    }
}

void GrammarSession::observe(bool timings, double prepare_seconds) noexcept {
    impl_->observed.timings_collected = timings;
    impl_->observed.prepare_seconds   = timings ? prepare_seconds : 0.0;
}

void GrammarSession::uploaded(std::size_t bytes) noexcept {
    impl_->observed.mask_upload_bytes += bytes;
}

const std::string* GrammarSession::budget_failure() const noexcept {
    return impl_->failure ? &*impl_->failure : nullptr;
}

ConstraintObservation GrammarSession::observation() const {
    if (impl_->tentative) throw std::logic_error("cannot publish tentative grammar state");
    auto result = impl_->observed;
    if (impl_->interrupted) return result; // the matcher is not consulted after an interruption
    result.terminated = impl_->matcher.IsTerminated();
    result.complete   = result.terminated || impl_->matcher.IsCompleted();
    return result;
}

void GrammarSession::confirm() noexcept {
    impl_->tentative = 0;
    if (impl_->state) impl_->state->confirm();
}

void GrammarSession::discard() {
    WorkTimer timer(impl_->observed.timings_collected, impl_->observed.matcher_seconds);
    if (impl_->tentative && !impl_->interrupted) { impl_->matcher.Rollback(impl_->tentative); }
    impl_->tentative = 0;
    if (impl_->state) impl_->state->discard();
}
} // namespace ninfer::text
