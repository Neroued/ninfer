#include "json_string_grammar.h"

#include <picojson.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "grammar_functor.h"
#include "fsm_builder.h"
#include <tuple>
#include "regex_converter.h"
#include "support/encoding.h"
#include "support/json_parse.h"

namespace xgrammar {
namespace {
std::string encode(const std::string& value) {
  auto json = picojson::value(value).serialize(false);
  return json.substr(1, json.size() - 2);
}

class JSONEncoder : public GrammarMutator {
  int32_t VisitByteString(const GrammarExpr& expr) final {
    std::string value;
    for (int byte : expr) value += static_cast<char>(byte);
    return builder_->AddByteString(encode(value));
  }

  int32_t VisitCharacterClass(const GrammarExpr& expr) final {
    std::vector<std::pair<int32_t, int32_t>> ranges;
    for (int i = 1; i < expr.size(); i += 2) ranges.emplace_back(expr[i], expr[i + 1]);
    if (expr[0]) {
      std::sort(ranges.begin(), ranges.end());
      std::vector<std::pair<int32_t, int32_t>> complement;
      int next = 0;
      for (const auto& [lo, hi] : ranges) {
        if (next < lo) complement.emplace_back(next, lo - 1);
        next = std::max(next, hi + 1);
      }
      if (next <= 0x10ffff) complement.emplace_back(next, 0x10ffff);
      ranges = std::move(complement);
    }
    std::vector<GrammarBuilder::CharacterClassElement> raw;
    std::vector<int32_t> choices;
    // ASCII control bytes, quote and backslash need JSON escapes. Surrogates are not scalar values.
    const std::pair<int, int> safe[] = {
        {0x20, 0x21}, {0x23, 0x5b}, {0x5d, 0xd7ff}, {0xe000, 0x10ffff}};
    for (const auto& [lo, hi] : ranges) {
      for (const auto& [a, b] : safe)
        if (std::max(lo, a) <= std::min(hi, b)) raw.emplace_back(std::max(lo, a), std::min(hi, b));
      for (int c = std::max(lo, 0); c <= std::min(hi, 0x5c); ++c) {
        if (c < 0x20 || c == '"' || c == '\\')
          choices.push_back(builder_->AddByteString(encode(std::string(1, static_cast<char>(c)))));
      }
    }
    if (!raw.empty()) choices.push_back(builder_->AddCharacterClass(raw));
    if (choices.empty()) return builder_->AddCharacterClass({{0, 0x10ffff}}, true);
    return choices.size() == 1 ? choices[0] : builder_->AddChoices(choices);
  }

  int32_t VisitCharacterClassStar(const GrammarExpr& expr) final {
    return builder_->AddRepeatFromExpr("json_char", VisitCharacterClass(expr), 0, -1);
  }
};
}  // namespace

int32_t AddScalarStringFSM(GrammarBuilder& builder, const FSMWithStartEnd& fsm,
                           const std::string& rule_name, bool close_json_string,
                           bool share_continuations) {
  // Do not emit paths that can never complete after exclusions. Otherwise a matcher
  // could accept a forbidden alternative's prefix and reach an all-rejected mask later.
  std::vector<std::vector<int>> predecessors(fsm.NumStates());
  for (int state = 0; state < fsm.NumStates(); ++state) {
    for (const auto& edge : fsm.GetFsm().GetEdges(state)) {
      predecessors[edge.target].push_back(state);
    }
  }
  std::vector<bool> productive(fsm.NumStates(), false);
  std::vector<int> worklist(fsm.GetEnds().begin(), fsm.GetEnds().end());
  for (int state : worklist) productive[state] = true;
  for (size_t index = 0; index < worklist.size(); ++index) {
    for (int state : predecessors[worklist[index]]) {
      if (!productive[state]) {
        productive[state] = true;
        worklist.push_back(state);
      }
    }
  }
  if (!productive[fsm.GetStart()]) {
    return builder.AddCharacterClass({{0, 0x10ffff}}, true);
  }

  // Collapse UTF-8 paths into codepoint transitions before emitting character classes.
  // Emitting individual continuation bytes as string literals would not round-trip through
  // EBNF, whose escaped string literals represent Unicode codepoints rather than raw bytes.
  struct CodepointRange {
    int min, max, target;
  };
  using Ranges = std::vector<CodepointRange>;
  auto merge_ranges = [](Ranges ranges) {
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
      return std::tie(a.target, a.min, a.max) < std::tie(b.target, b.min, b.max);
    });
    Ranges merged;
    for (const auto& range : ranges) {
      if (!merged.empty() && merged.back().target == range.target &&
          range.min <= merged.back().max + 1) {
        merged.back().max = std::max(merged.back().max, range.max);
      } else {
        merged.push_back(range);
      }
    }
    return merged;
  };
  auto append_ranges = [](Ranges* ranges, int min, int max, int shift,
                          const CodepointRange& suffix) {
    if (suffix.min == 0 && suffix.max == (1 << shift) - 1) {
      ranges->push_back({min << shift, (max << shift) | suffix.max, suffix.target});
    } else {
      for (int prefix = min; prefix <= max; ++prefix) {
        ranges->push_back(
            {(prefix << shift) | suffix.min, (prefix << shift) | suffix.max, suffix.target});
      }
    }
  };
  // Memoize suffixes so wide Unicode classes do not enumerate every codepoint.
  std::map<std::pair<int, int>, Ranges> suffix_cache;
  std::function<const Ranges&(int, int)> suffix_ranges = [&](int state,
                                                             int remaining) -> const Ranges& {
    auto [it, inserted] = suffix_cache.emplace(std::make_pair(state, remaining), Ranges{});
    if (!inserted) return it->second;
    Ranges ranges;
    if (remaining == 0) {
      ranges.push_back({0, 0, state});
    } else {
      for (const auto& edge : fsm.GetFsm().GetEdges(state)) {
        if (!productive[edge.target]) continue;
        XGRAMMAR_CHECK(edge.IsCharRange());
        int min = std::max(edge.min, 0x80), max = std::min(edge.max, 0xbf);
        if (min > max) continue;
        for (const auto& suffix : suffix_ranges(edge.target, remaining - 1)) {
          append_ranges(&ranges, min & 0x3f, max & 0x3f, 6 * (remaining - 1), suffix);
        }
      }
    }
    it->second = merge_ranges(std::move(ranges));
    return it->second;
  };

  std::vector<int32_t> rules(fsm.NumStates(), -1);
  std::vector<int> pending{fsm.GetStart()};
  // Share complete fallback branches across key prefixes to reuse their token masks.
  // The key contains the target state followed by every codepoint interval.
  std::map<std::vector<int32_t>, int32_t> shared_key_continuations;
  rules[fsm.GetStart()] = builder.AddEmptyRuleWithHint(rule_name + "_exclude");
  for (size_t index = 0; index < pending.size(); ++index) {
    int state = pending[index];
    Ranges ranges;
    for (const auto& edge : fsm.GetFsm().GetEdges(state)) {
      if (!productive[edge.target]) continue;
      XGRAMMAR_CHECK(edge.IsCharRange());
      if (edge.min < 128) {
        ranges.push_back({edge.min, std::min(edge.max, 127), edge.target});
      }
      const int leading_min[] = {0, 0xc2, 0xe0, 0xf0};
      const int leading_max[] = {0, 0xdf, 0xef, 0xf4};
      for (int remaining = 1; remaining <= 3; ++remaining) {
        int min = std::max(edge.min, leading_min[remaining]);
        int max = std::min(edge.max, leading_max[remaining]);
        if (min > max) continue;
        int mask = (1 << (6 - remaining)) - 1;
        Ranges unicode_ranges;
        for (const auto& suffix : suffix_ranges(edge.target, remaining)) {
          append_ranges(&unicode_ranges, min & mask, max & mask, 6 * remaining, suffix);
        }
        // The byte FSM can contain non-canonical UTF-8 paths. Never turn an overlong
        // encoding into a second transition for an ASCII character (bypassing exclusions).
        const int codepoint_min[] = {0, 0x80, 0x800, 0x10000};
        const int codepoint_max[] = {0, 0x7ff, 0xffff, 0x10ffff};
        for (auto range : unicode_ranges) {
          range.min = std::max(range.min, codepoint_min[remaining]);
          range.max = std::min(range.max, codepoint_max[remaining]);
          if (range.min <= range.max) {
            // Raw tool strings and JSON strings contain Unicode scalar values.
            if (range.min <= 0xd7ff)
              ranges.push_back({range.min, std::min(range.max, 0xd7ff), range.target});
            if (range.max >= 0xe000)
              ranges.push_back({std::max(range.min, 0xe000), range.max, range.target});
          }
        }
      }
    }
    std::map<int, std::vector<GrammarBuilder::CharacterClassElement>> transitions;
    for (const auto& range : merge_ranges(std::move(ranges))) {
      transitions[range.target].push_back({range.min, range.max});
    }
    std::vector<int32_t> choices;
    // Keep the closing quote on the accepting body states. A nullable body rule
    // would otherwise finish at every character and force token masks to speculate
    // across its parent rules. The quote is appended AFTER the exclusion intersection:
    // it is JSON syntax, not string content subject to excludes.
    if (fsm.IsEndState(state)) {
      choices.push_back(close_json_string ? builder.AddByteString("\"") : builder.AddEmptyStr());
    }
    for (const auto& [target, codepoints] : transitions) {
      if (rules[target] == -1) {
        rules[target] = builder.AddEmptyRuleWithHint(rule_name + "_exclude");
        pending.push_back(target);
      }
      bool single_ascii = codepoints.size() == 1 && codepoints[0].lower == codepoints[0].upper &&
                          codepoints[0].upper <= 0x7f;
      // Keep self-loops local for the compiler's speculative string-mask fast path.
      if (share_continuations && !single_ascii && target != state) {
        std::vector<int32_t> key{target};
        key.reserve(1 + codepoints.size() * 2);
        for (const auto& range : codepoints) {
          key.push_back(range.lower);
          key.push_back(range.upper);
        }
        auto [it, inserted] = shared_key_continuations.emplace(std::move(key), -1);
        if (inserted) {
          it->second =
              builder.AddRuleWithHint(rule_name + "_exclude_continuation",
                                      builder.AddSequence({builder.AddCharacterClass(codepoints),
                                                           builder.AddRuleRef(rules[target])}));
        }
        choices.push_back(builder.AddRuleRef(it->second));
      } else {
        choices.push_back(builder.AddSequence(
            {builder.AddCharacterClass(codepoints), builder.AddRuleRef(rules[target])}));
      }
    }
    if (choices.empty()) {
      choices.push_back(builder.AddCharacterClass({{0, 0x10ffff}}, true));
    }
    builder.UpdateRuleBody(rules[state], builder.AddChoices(choices));
  }
  return builder.AddRuleRef(rules[fsm.GetStart()]);
}

Grammar StringConstraints(const std::vector<std::string>& patterns, int minimum, int maximum,
                          const std::vector<std::string>& excluded, bool json_encoding) {
  if (minimum > 8192 || maximum > 8192)
    throw JSONSchemaCompileError(
        SchemaErrorType::kUnsupportedSchema,
        "string conjunction length exceeds the 8192-character compilation limit");
  const std::string length = R"([^\uD800-\uDFFF])" + std::string("{") + std::to_string(minimum) +
                             "," + (maximum < 0 ? "" : std::to_string(maximum)) + "}";
  auto initial = GrammarFSMBuilder::Regex(length, false);
  if (initial.IsErr()) throw std::invalid_argument(std::move(initial).UnwrapErr().what());
  auto fsm = std::move(initial).Unwrap();
  const auto intersect = [&](const FSMWithStartEnd& other) {
    auto joined = FSMWithStartEnd::Intersect(fsm, other);
    if (joined.IsErr()) throw std::invalid_argument(std::move(joined).UnwrapErr().what());
    fsm = std::move(joined).Unwrap();
    if (fsm.NumStates() > 65536)
      throw JSONSchemaCompileError(SchemaErrorType::kUnsupportedSchema,
                                   "string constraint exceeds 65536 compiled states");
  };
  for (const auto& pattern : patterns) {
    auto parsed = GrammarFSMBuilder::Regex(SchemaStringPattern(pattern), false);
    if (parsed.IsErr()) throw std::invalid_argument(std::move(parsed).UnwrapErr().what());
    intersect(std::move(parsed).Unwrap());
  }
  auto exclusion = GrammarFSMBuilder::TagDispatch({{}, false, excluded});
  if (!exclusion) throw std::invalid_argument("invalid string exclusion");
  intersect(*exclusion);
  GrammarBuilder builder;
  const auto body = AddScalarStringFSM(builder, fsm, "value", false, false);
  auto grammar = builder.Get(builder.AddRuleWithHint("root", body));
  if (json_encoding) grammar = JSONEncoder().Apply(grammar);
  return GrammarNormalizer::Apply(grammar);
}

std::string SchemaStringPattern(const std::string& pattern) {
  return NormalizeRegexPattern(pattern, false);
}

Grammar JSONStringPattern(const std::string& pattern) {
  return GrammarNormalizer::Apply(
      JSONEncoder().Apply(Grammar::FromRegex(SchemaStringPattern(pattern))));
}

Grammar JSONStringLength(int minimum, int maximum) {
  std::string count = "{" + std::to_string(minimum) + ",";
  if (maximum >= 0) count += std::to_string(maximum);
  count += "}";
  return GrammarNormalizer::Apply(JSONEncoder().Apply(Grammar::FromEBNF("root ::= [^]" + count)));
}
Grammar JSONStringExcept(const std::vector<std::string>& excluded) {
  struct Node {
    bool terminal = false;
    std::map<int, Node> next;
  } root;
  for (const auto& key : excluded) {
    Node* node = &root;
    for (size_t offset = 0; offset < key.size();) {
      const auto [cp, length] = ParseNextUTF8(key.c_str() + offset);
      XGRAMMAR_CHECK(length > 0) << "invalid property name UTF-8";
      node = &node->next[cp];
      offset += length;
    }
    node->terminal = true;
  }
  GrammarBuilder builder;
  const auto any_char = builder.AddCharacterClass({{0, 0x10ffff}});
  const auto any = builder.AddRepeatFromExpr("tail", any_char, 0, -1);
  const std::function<int32_t(const Node&)> build = [&](const Node& node) {
    std::vector<int32_t> alternatives;
    if (!node.terminal) alternatives.push_back(builder.AddEmptyStr());
    std::vector<GrammarBuilder::CharacterClassElement> excluded_chars;
    for (const auto& [cp, next] : node.next) excluded_chars.push_back({cp, cp});
    alternatives.push_back(
        builder.AddSequence({builder.AddCharacterClass(excluded_chars, true), any}));
    for (const auto& [cp, next] : node.next) {
      auto ref = builder.AddRuleRef(builder.AddRuleWithHint("key", build(next)));
      alternatives.push_back(builder.AddSequence({builder.AddByteString(CharToUTF8(cp)), ref}));
    }
    return builder.AddChoices(alternatives);
  };
  const auto root_id = builder.AddRule("root", build(root));
  return GrammarNormalizer::Apply(JSONEncoder().Apply(builder.Get(root_id)));
}
}  // namespace xgrammar
