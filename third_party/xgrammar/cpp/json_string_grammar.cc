#include "json_string_grammar.h"

#include <picojson.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <utility>

#include "grammar_functor.h"
#include "fsm_builder.h"
#include "regex_converter.h"
#include "support/encoding.h"
#include "support/json_parse.h"

namespace xgrammar {
namespace {
std::string encode(const std::string& value) {
  auto json = picojson::value(value).serialize(false);
  return json.substr(1, json.size() - 2);
}

FSMWithStartEnd Minimize(FSMWithStartEnd fsm) {
  auto result = fsm.MinimizeDFA(65536);
  if (result.IsErr()) throw std::move(result).UnwrapErr();
  return std::move(result).Unwrap();
}

void AddBytes(FSM& fsm, int source, int target, const std::string& bytes) {
  for (size_t i = 0; i < bytes.size(); ++i) {
    const int next = i + 1 == bytes.size() ? target : fsm.AddState();
    const auto byte = static_cast<unsigned char>(bytes[i]);
    fsm.AddEdge(source, next, byte, byte);
    source = next;
  }
}

FSMWithStartEnd JSONEncodedFSM(const FSMWithStartEnd& input) {
  FSM result;
  for (int i = 0; i < input.NumStates(); ++i) result.AddState();
  for (int state = 0; state < input.NumStates(); ++state) {
    for (const auto& edge : input.GetFsm().GetEdges(state)) {
      // UTF-8 multibyte sequences are unchanged. Escape only JSON's ASCII syntax/control bytes.
      for (const auto [lo, hi] : {std::pair{0x20, 0x21}, {0x23, 0x5b}, {0x5d, 0xff}})
        if (std::max(lo, edge.min) <= std::min(hi, edge.max))
          result.AddEdge(state, edge.target, std::max(lo, edge.min), std::min(hi, edge.max));
      for (int byte = edge.min; byte <= std::min(edge.max, 0x5c); ++byte)
        if (byte < 0x20 || byte == '"' || byte == '\\')
          AddBytes(result, state, edge.target, encode(std::string(1, static_cast<char>(byte))));
    }
  }
  return Minimize(FSMWithStartEnd(result, input.GetStart(), input.GetEnds()));
}

class JSONEncoder : public GrammarMutator {
  int32_t VisitByteFSM(const GrammarExpr& expr) final {
    return builder_->AddByteFSM(JSONEncodedFSM(base_grammar_->GetByteFSM(expr)));
  }

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

int32_t AddScalarStringFSM(GrammarBuilder& builder, const FSMWithStartEnd& input,
                           const std::string& rule_name, bool close_json_string) {
  // Preserve Unicode scalar semantics, including where byte regexes admit overlong encodings.
  static const auto scalar = [] {
    auto parsed = GrammarFSMBuilder::Regex(R"([^\uD800-\uDFFF]*)", false);
    if (parsed.IsErr()) throw std::move(parsed).UnwrapErr();
    return Minimize(std::move(parsed).Unwrap());
  }();
  auto intersection = FSMWithStartEnd::Intersect(input, scalar);
  if (intersection.IsErr()) throw std::move(intersection).UnwrapErr();
  auto fsm = Minimize(std::move(intersection).Unwrap());
  if (close_json_string) {
    const auto ends = fsm.GetEnds();
    const int end = fsm.AddState();
    for (int old : ends) fsm.GetFsm().AddEdge(old, end, '"', '"');
    fsm = FSMWithStartEnd(fsm.GetFsm(), fsm.GetStart(), {end});
  }
  return builder.AddRuleRef(builder.AddRuleWithHint(rule_name, builder.AddByteFSM(fsm)));
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
  const auto body = AddScalarStringFSM(builder, fsm, "value", false);
  auto grammar = builder.Get(builder.AddRuleWithHint("root", body));
  if (json_encoding) grammar = JSONEncoder().Apply(grammar);
  return GrammarNormalizer::Apply(grammar);
}

StringDispatch::StringDispatch(GrammarBuilder& builder, std::string delimiter,
                               int32_t default_value, std::string forbidden_ascii, bool nonempty)
    : builder_(builder), delimiter_(std::move(delimiter)), default_value_(default_value) {
  XGRAMMAR_CHECK(!delimiter_.empty());
  std::string allowed = "[^\\uD800-\\uDFFF";
  for (unsigned char byte : forbidden_ascii) {
    XGRAMMAR_CHECK(byte < 128);
    constexpr char hex[] = "0123456789abcdef";
    allowed += "\\x";
    allowed += hex[byte >> 4];
    allowed += hex[byte & 15];
  }
  allowed += nonempty ? "]+" : "]*";
  auto raw = GrammarFSMBuilder::Regex(allowed, false);
  if (raw.IsErr()) throw std::move(raw).UnwrapErr();
  auto stop = GrammarFSMBuilder::TagDispatch({{}, false, {delimiter_}});
  XGRAMMAR_CHECK(stop.has_value());
  auto body = FSMWithStartEnd::Intersect(std::move(raw).Unwrap(), *stop);
  if (body.IsErr()) throw std::move(body).UnwrapErr();
  auto fsm = std::move(body).Unwrap();
  const auto ends = fsm.GetEnds();
  const int final = fsm.AddState();
  for (int end : ends) AddBytes(fsm.GetFsm(), end, final, delimiter_);
  lexer_ = Minimize(FSMWithStartEnd(fsm.GetFsm(), fsm.GetStart(), {final}));
}

int32_t StringDispatch::Add(const std::vector<std::pair<std::string, int32_t>>& cases) {
  struct Node {
    std::map<unsigned char, size_t> children;
    int state;
    bool terminal = false;
  };
  const auto advance = [&](int state, unsigned char byte) {
    for (const auto& edge : lexer_.GetFsm().GetEdges(state))
      if (edge.min <= byte && byte <= edge.max) return edge.target;
    return -1;
  };
  std::vector<Node> nodes{{{}, lexer_.GetStart()}};
  std::map<int32_t, std::vector<std::string>> values;
  for (const auto& [name, value] : cases) {
    const auto full = name + delimiter_;
    size_t node = 0;
    for (unsigned char byte : full) {
      auto found = nodes[node].children.find(byte);
      if (found != nodes[node].children.end()) {
        node = found->second;
      } else {
        const int state = advance(nodes[node].state, byte);
        XGRAMMAR_CHECK(state >= 0) << "Name is outside the delimited string language";
        const size_t child = nodes.size();
        nodes[node].children.emplace(byte, child);
        nodes.push_back({{}, state});
        node = child;
      }
    }
    XGRAMMAR_CHECK(lexer_.IsEndState(nodes[node].state) && !nodes[node].terminal);
    nodes[node].terminal = true;
    values[value].push_back(full);
  }
  const auto branch = [&](const FSMWithStartEnd& fsm, int32_t value) {
    const int name = builder_.AddRuleWithHint("parameter_name", builder_.AddByteFSM(fsm));
    return builder_.AddSequence({builder_.AddRuleRef(name), builder_.AddRuleRef(value)});
  };
  std::vector<int32_t> choices;
  for (const auto& [value, names] : values) {
    auto trie = TrieFSMBuilder::Build(names, {});
    XGRAMMAR_CHECK(trie.has_value());
    choices.push_back(branch(Minimize(*trie), value));
  }
  // A mismatch leaves the positive trie once and joins the shared name lexer.
  // Do not compose a regex complement or replicate the suffix at each prefix.
  FSM fallback;
  const int suffix = nodes.size();
  for (int i = 0; i < suffix + lexer_.NumStates(); ++i) fallback.AddState();
  std::vector<int32_t> ends;
  for (int i = 0; i < suffix; ++i) {
    const auto& node = nodes[i];
    if (!node.terminal && lexer_.IsEndState(node.state)) ends.push_back(i);
    int first = 0, previous = -1;
    for (int byte = 0; byte <= 256; ++byte) {
      int target = -1;
      if (byte < 256) {
        const int next = advance(node.state, static_cast<unsigned char>(byte));
        if (next >= 0) {
          auto child = node.children.find(byte);
          target = child == node.children.end() ? suffix + next : int(child->second);
        }
      }
      if (target != previous) {
        if (previous >= 0) fallback.AddEdge(i, previous, first, byte - 1);
        first = byte;
        previous = target;
      }
    }
  }
  for (int state = 0; state < lexer_.NumStates(); ++state) {
    if (lexer_.IsEndState(state)) ends.push_back(suffix + state);
    for (const auto& edge : lexer_.GetFsm().GetEdges(state))
      fallback.AddEdge(suffix + state, suffix + edge.target, edge.min, edge.max);
  }
  choices.push_back(branch(Minimize(FSMWithStartEnd(fallback, 0, ends)), default_value_));
  return builder_.AddRuleWithHint("parameter_dispatch", builder_.AddChoices(choices));
}

std::string SchemaStringPattern(const std::string& pattern) {
  return NormalizeRegexPattern(pattern, false);
}

Grammar JSONStringPattern(const std::string& pattern) {
  return GrammarNormalizer::Apply(
      JSONEncoder().Apply(Grammar::FromRegex(SchemaStringPattern(pattern))));
}

Grammar JSONStringLength(int minimum, int maximum) {
  auto scalar = JSONEncoder().Apply(Grammar::FromEBNF("root ::= [^]"));
  GrammarBuilder builder;
  const int32_t unit = SubGrammarAdder::Apply(&builder, scalar);
  const int32_t body =
      builder.AddSequence({builder.AddByteString("\""), builder.AddRepeat(unit, minimum, maximum),
                           builder.AddByteString("\"")});
  return GrammarNormalizer::Apply(builder.Get(builder.AddRuleWithHint("string", body)));
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
