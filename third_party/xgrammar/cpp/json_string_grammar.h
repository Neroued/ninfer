#ifndef XGRAMMAR_JSON_STRING_GRAMMAR_H_
#define XGRAMMAR_JSON_STRING_GRAMMAR_H_

#include <xgrammar/grammar.h>
#include "fsm.h"
#include "grammar_builder.h"

#include <string>
#include <utility>
#include <vector>

namespace xgrammar {
int32_t AddScalarStringFSM(GrammarBuilder& builder, const FSMWithStartEnd& fsm,
                           const std::string& rule_name, bool close_json_string);
Grammar StringConstraints(const std::vector<std::string>& patterns, int minimum, int maximum,
                          const std::vector<std::string>& excluded, bool json_encoding);

// Dispatch an exact delimited name to its value rule. All unmatched names share one
// suffix automaton; cases form a positive trie, with no per-case complement language.
class StringDispatch {
 public:
  StringDispatch(GrammarBuilder& builder, std::string delimiter, int32_t default_value,
                 std::string forbidden_ascii = {}, bool nonempty = false);
  int32_t Add(const std::vector<std::pair<std::string, int32_t>>& cases);

 private:
  GrammarBuilder& builder_;
  std::string delimiter_;
  int32_t default_value_;
  FSMWithStartEnd lexer_;
};

// JSON-encoded string body matching a schema pattern; the caller supplies quotes.
Grammar JSONStringPattern(const std::string& pattern);
// Search semantics and ECMAScript character classes, before JSON string encoding.
std::string SchemaStringPattern(const std::string& pattern);
// Complete JSON string, including quotes, bounded by decoded Unicode scalar count.
Grammar JSONStringLength(int minimum, int maximum);
Grammar JSONStringExcept(const std::vector<std::string>& excluded);
}  // namespace xgrammar
#endif
