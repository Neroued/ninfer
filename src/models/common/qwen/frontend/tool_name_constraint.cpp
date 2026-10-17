#include "models/common/qwen/frontend/tool_name_constraint.h"
#include "models/common/qwen/frontend/tool_format.h"

#include <unordered_map>
#include <unordered_set>

namespace ninfer::models::qwen::frontend {
namespace {

// Static grammar owns syntax and value encoding. This companion recognizes only the canonical
// outer framing to enforce unique top-level names, including inside a single candidate token.
class ToolNameConstraint final : public text::GrammarStateConstraint {
    enum class Phase {
        Reasoning,
        Branch,
        Text,
        Function,
        Parameters,
        Name,
        ValueStart,
        Value,
        ToolClose,
        Body
    };

    struct Cursor {
        Phase phase         = Phase::Text;
        std::size_t matched = 0, closing = 0, call = 0, name_begin = 0;
    };

    struct Inserted {
        std::size_t call;
        const std::string* name;
    };

    struct Mark {
        Cursor cursor;
        std::size_t bytes, inserted;
    };

public:
    ToolNameConstraint(std::shared_ptr<const text::GrammarVocabulary> vocabulary,
                       std::string_view close, bool combined)
        : vocabulary_(std::move(vocabulary)), endings_(vocabulary_->containing('>')),
          reasoning_close_(close), combined_(combined) {
        cursor_.phase = !close.empty() ? Phase::Reasoning : body_phase();
        committed_    = mark();
    }

    void filter(std::span<std::uint32_t> mask) override {
        if (cursor_.phase == Phase::Body) return;
        const auto before = mark();
        for (const auto id : endings_) {
            if (!(mask[id / 32] & (1u << (id % 32)))) continue;
            const auto& bytes = vocabulary_->pieces()[id];
            // Completing a name needs '>'. Outside an already-started header it also needs the
            // entire parameter opener in this token. Most value tokens need no simulation.
            if (cursor_.phase != Phase::Name &&
                !(cursor_.phase == Phase::Parameters && cursor_.matched) &&
                bytes.find(kParamOpen) == std::string::npos)
                continue;
            try {
                if (!consume(bytes)) mask[id / 32] &= ~(1u << (id % 32));
            } catch (...) {
                restore(before);
                throw;
            }
            restore(before);
        }
    }

    bool accept(std::string_view bytes) override {
        const auto before = mark();
        try {
            if (consume(bytes)) return true;
        } catch (...) {
            restore(before);
            throw;
        }
        restore(before);
        return false;
    }

    void confirm() noexcept override {
        inserted_.clear();
        for (auto it = seen_.begin(); it != seen_.end();) {
            if (it->first != cursor_.call)
                it = seen_.erase(it);
            else
                ++it;
        }
        if (cursor_.phase == Phase::Name)
            names_.erase(0, cursor_.name_begin);
        else
            names_.clear();
        cursor_.name_begin = 0;
        committed_         = mark();
    }

    void discard() noexcept override { restore(committed_); }

private:
    Phase body_phase() const { return combined_ ? Phase::Branch : Phase::Text; }

    Mark mark() const { return {cursor_, names_.size(), inserted_.size()}; }

    void restore(const Mark& mark) noexcept {
        while (inserted_.size() > mark.inserted) {
            const auto value = inserted_.back();
            auto found       = seen_.find(value.call);
            found->second.erase(*value.name);
            if (found->second.empty()) seen_.erase(found);
            inserted_.pop_back();
        }
        names_.resize(mark.bytes);
        cursor_ = mark.cursor;
    }

    bool consume(std::string_view bytes) {
        for (const char byte : bytes) {
            switch (cursor_.phase) {
            case Phase::Reasoning:
                if (match_tool_delimiter(reasoning_close_, cursor_.matched, byte))
                    cursor_.phase = body_phase();
                break;
            case Phase::Branch:
                // JSON and tool output have disjoint first non-whitespace characters.
                if (byte == ' ' || byte == '\n' || byte == '\r' || byte == '\t') break;
                cursor_.phase = byte == '<' ? Phase::Text : Phase::Body;
                if (cursor_.phase == Phase::Body) break;
                [[fallthrough]];
            case Phase::Text:
                if (match_tool_delimiter(kToolOpen, cursor_.matched, byte)) {
                    ++cursor_.call;
                    cursor_.phase = Phase::Function;
                }
                break;
            case Phase::Function:
                if (byte == '>') cursor_.phase = Phase::Parameters;
                break;
            case Phase::Parameters:
                if (match_tool_delimiter(kParamOpen, cursor_.matched, byte)) {
                    cursor_.phase      = Phase::Name;
                    cursor_.closing    = 0;
                    cursor_.name_begin = names_.size();
                } else if (match_tool_delimiter(kFunctionClose, cursor_.closing, byte)) {
                    cursor_.phase   = Phase::ToolClose;
                    cursor_.matched = 0;
                }
                break;
            case Phase::Name:
                if (byte != '>') {
                    names_ += byte;
                } else {
                    auto& names               = seen_[cursor_.call];
                    const auto [entry, added] = names.emplace(names_.substr(cursor_.name_begin));
                    if (!added) return false;
                    inserted_.push_back({cursor_.call, &*entry});
                    cursor_.phase = Phase::ValueStart;
                }
                break;
            case Phase::ValueStart:
                // The opening framing newline is not part of the value or closing delimiter.
                cursor_.phase = Phase::Value;
                break;
            case Phase::Value:
                // Canonical raw strings exclude this delimiter. Valid JSON cannot contain a
                // literal newline followed by markup in a string or outside its JSON syntax.
                if (match_tool_delimiter(kParameterEnd, cursor_.matched, byte))
                    cursor_.phase = Phase::Parameters;
                break;
            case Phase::ToolClose:
                if (match_tool_delimiter(kToolClose, cursor_.matched, byte))
                    cursor_.phase = Phase::Text;
                break;
            case Phase::Body:
                break;
            }
        }
        return true;
    }

    std::shared_ptr<const text::GrammarVocabulary> vocabulary_;
    const std::vector<std::int32_t>& endings_;
    std::string reasoning_close_;
    bool combined_;
    Cursor cursor_;
    std::string names_;
    std::unordered_map<std::size_t, std::unordered_set<std::string>> seen_;
    std::vector<Inserted> inserted_;
    Mark committed_;
};
} // namespace

std::unique_ptr<text::GrammarStateConstraint>
make_tool_name_constraint(std::shared_ptr<const text::GrammarVocabulary> vocabulary,
                          std::string_view reasoning_close, bool combined_body) {
    return std::make_unique<ToolNameConstraint>(std::move(vocabulary), reasoning_close,
                                                combined_body);
}
} // namespace ninfer::models::qwen::frontend
