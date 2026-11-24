/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/regex_converter.cc
 */
#include "regex_converter.h"

#include <iostream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "support/encoding.h"
#include "support/logging.h"
#include "support/utils.h"

namespace xgrammar {

/*!
 * \brief Convert a regex to EBNF.
 * \details The implementation refers to the regex described in
 * https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_expressions
 */
class RegexConverter {
 public:
  explicit RegexConverter(const std::string& regex) : regex_(regex) {
    if (!regex.empty()) {
      // ParseUTF8 takes a C string and stops at the first NUL, so a regex
      // containing one would either yield an empty codepoint vector (leading
      // NUL -- an out-of-bounds read below) or be silently truncated to the
      // prefix. Product normalization represents literal NUL as an escape before this parser.
      if (regex.find('\0') != std::string::npos) {
        XGRAMMAR_LOG(FATAL) << "The regex must not contain null characters.";
        XGRAMMAR_UNREACHABLE();
      }
      regex_codepoints_ = ParseUTF8(regex_.c_str(), false);
      if (regex_codepoints_[0] == kInvalidUTF8) {
        XGRAMMAR_LOG(FATAL) << "The regex is not a valid UTF-8 string.";
        XGRAMMAR_UNREACHABLE();
      }
    }
    regex_codepoints_.push_back(0);  // Add a null terminator
  }
  std::string Convert();

 private:
  /**
   * \brief Add a segment string to the result EBNF string. It especially adds a space if needed
   * and add_space is true.
   */
  void AddEBNFSegment(const std::string& element);

  [[noreturn]] void RaiseError(const std::string& message);
  void RaiseWarning(const std::string& message);

  std::string HandleCharacterClass();
  std::string HandleRepetitionRange();
  std::string HandleCharEscape();
  std::string HandleEscape();
  std::string HandleEscapeInCharClass();
  /**
   * \brief Handle group modifier. The general format is "(?" + modifier + content + ")". E.g.
   * "(?:abc)" is a non-capturing group.
   */
  void HandleGroupModifier();

  std::string regex_;
  std::vector<TCodepoint> regex_codepoints_;
  TCodepoint* start_;
  TCodepoint* current_;
  TCodepoint* end_;
  std::string result_ebnf_;
  int parenthesis_level_ = 0;
};

void RegexConverter::AddEBNFSegment(const std::string& element) {
  if (!result_ebnf_.empty()) {
    result_ebnf_ += ' ';
  }
  result_ebnf_ += element;
}

void RegexConverter::RaiseError(const std::string& message) {
  XGRAMMAR_LOG(FATAL) << "Regex parsing error at position " << current_ - start_ + 1 << ": "
                      << message;
  XGRAMMAR_UNREACHABLE();
}

void RegexConverter::RaiseWarning(const std::string& message) {
  XGRAMMAR_LOG(WARNING) << "Regex parsing warning at position " << current_ - start_ + 1 << ": "
                        << message;
}

std::string RegexConverter::HandleCharacterClass() {
  std::string char_class = "[";
  ++current_;
  if (*current_ == ']') {
    RaiseError("Empty character class is not allowed in regex.");
  }
  while (*current_ != ']' && current_ != end_) {
    if (*current_ == '\\') {
      char_class += HandleEscapeInCharClass();
    } else {
      char_class += CharToUTF8(*current_);
      ++current_;
    }
  }
  if (current_ == end_) {
    RaiseError("Unclosed '['");
  }
  char_class += ']';
  ++current_;
  return char_class;
}

// {x}: Match exactly x occurrences of the preceding regular expression.
// {x,}
// {x,y}
std::string RegexConverter::HandleRepetitionRange() {
  std::string result = "{";
  ++current_;
  if (!isdigit(*current_)) {
    RaiseError("Invalid repetition count.");
  }
  while (isdigit(*current_)) {
    result += static_cast<char>(*current_);
    ++current_;
  }
  if (*current_ != ',' && *current_ != '}') {
    RaiseError("Invalid repetition count.");
  }
  result += static_cast<char>(*current_);
  ++current_;
  if (current_[-1] == '}') {
    // Matches {x}
    return result;
  }
  if (!isdigit(*current_) && *current_ != '}') {
    RaiseError("Invalid repetition count.");
  }
  while (isdigit(*current_)) {
    result += static_cast<char>(*current_);
    ++current_;
  }
  if (*current_ != '}') {
    RaiseError("Invalid repetition count.");
  }
  result += '}';
  ++current_;
  return result;
}

std::string RegexConverter::HandleCharEscape() {
  // clang-format off
  static const std::unordered_map<char, TCodepoint> CUSTOM_ESCAPE_MAP = {
      {'^', '^'}, {'$', '$'}, {'.', '.'}, {'*', '*'}, {'+', '+'}, {'?', '?'}, {'\\', '\\'},
      {'(', '('}, {')', ')'}, {'[', '['}, {']', ']'}, {'{', '{'}, {'}', '}'}, {'|', '|'},
      {'/', '/'}, {'-', '-'}
  };
  // clang-format on
  if (end_ - current_ < 2 || (current_[1] == 'u' && end_ - current_ < 5) ||
      (current_[1] == 'x' && end_ - current_ < 4) || (current_[1] == 'c' && end_ - current_ < 3)) {
    RaiseError("Escape sequence is not finished.");
  }
  // Regex hex escapes have exactly two digits; the generic C/GBNF reader consumes all
  // following hex digits (e.g. it would read \\x41B as one character instead of "AB").
  if (current_[1] == 'x') {
    const int high = HexCharToInt(current_[2]), low = HexCharToInt(current_[3]);
    if (high < 0 || low < 0) RaiseError("Invalid hexadecimal escape sequence.");
    current_ += 4;
    return EscapeString(high * 16 + low);
  }
  auto [codepoint, len] = ParseNextEscaped(current_, CUSTOM_ESCAPE_MAP);
  if (codepoint != CharHandlingError::kInvalidEscape) {
    current_ += len;
    return EscapeString(codepoint);
  } else if (current_[1] == 'u' && current_[2] == '{') {
    current_ += 3;
    int len = 0;
    TCodepoint value = 0;
    while (HexCharToInt(current_[len]) != -1 && len <= 6) {
      value = value * 16 + HexCharToInt(current_[len]);
      ++len;
    }
    if (len == 0 || len > 6 || current_[len] != '}') {
      RaiseError("Invalid Unicode escape sequence.");
    }
    current_ += len + 1;
    return EscapeString(value);
  } else if (current_[1] == 'c') {
    current_ += 2;
    if (!std::isalpha(*current_)) {
      RaiseError("Invalid control character escape sequence.");
    }
    ++current_;
    return EscapeString((*(current_ - 1)) % 32);
  } else {
    RaiseWarning(
        "Escape sequence '\\" + EscapeString(current_[1]) +
        "' is not recognized. The character itself will be matched"
    );
    current_ += 2;
    return EscapeString(current_[-1]);
  }
}

std::string RegexConverter::HandleEscapeInCharClass() {
  if (end_ - current_ < 2) {
    RaiseError("Escape sequence is not finished.");
  }
  if (current_[1] == 'd') {
    current_ += 2;
    return "0-9";
  } else if (current_[1] == 'D') {
    current_ += 2;
    return R"(\x00-\x2F\x3A-\U0010FFFF)";
  } else if (current_[1] == 'w') {
    current_ += 2;
    return "a-zA-Z0-9_";
  } else if (current_[1] == 'W') {
    current_ += 2;
    return R"(\x00-\x2F\x3A-\x40\x5B-\x5E\x60\x7B-\U0010FFFF)";
  } else if (current_[1] == 's') {
    current_ += 2;
    return R"(\f\n\r\t\v\u0020\u00a0)";
  } else if (current_[1] == 'S') {
    current_ += 2;
    return R"(\x00-\x08\x0E-\x1F\x21-\x9F\xA1-\U0010FFFF)";
  } else {
    auto res = HandleCharEscape();
    if (res == "]" || res == "-" || res == "^" || res == "[") {
      return "\\" + res;
    } else {
      return res;
    }
  }
}

std::string RegexConverter::HandleEscape() {
  // clang-format off
  static const std::unordered_map<char, TCodepoint> CUSTOM_ESCAPE_MAP = {
      {'^', '^'}, {'$', '$'}, {'.', '.'}, {'*', '*'}, {'+', '+'}, {'?', '?'}, {'\\', '\\'},
      {'(', '('}, {')', ')'}, {'[', '['}, {']', ']'}, {'{', '{'}, {'}', '}'}, {'|', '|'},
      {'/', '/'}
  };
  // clang-format on
  if (end_ - current_ < 2) {
    RaiseError("Escape sequence is not finished.");
  }
  if (current_[1] == 'd') {
    current_ += 2;
    return "[0-9]";
  } else if (current_[1] == 'D') {
    current_ += 2;
    return "[^0-9]";
  } else if (current_[1] == 'w') {
    current_ += 2;
    return "[a-zA-Z0-9_]";
  } else if (current_[1] == 'W') {
    current_ += 2;
    return "[^a-zA-Z0-9_]";
  } else if (current_[1] == 's') {
    current_ += 2;
    return R"([\f\n\r\t\v\u0020\u00a0])";
  } else if (current_[1] == 'S') {
    current_ += 2;
    return R"([^\f\n\r\t\v\u0020\u00a0])";
  } else if ((current_[1] >= '1' && current_[1] <= '9') || current_[1] == 'k') {
    RaiseError("Backreference is not supported yet.");
  } else if (current_[1] == 'p' || current_[1] == 'P') {
    RaiseError("Unicode character class escape sequence is not supported yet.");
  } else if (current_[1] == 'b' || current_[1] == 'B') {
    RaiseError("Word boundary is not supported yet.");
  } else {
    return "\"" + HandleCharEscape() + "\"";
  }
}

void RegexConverter::HandleGroupModifier() {
  if (current_ == end_) {
    RaiseError("Group modifier is not finished.");
  }
  if (*current_ == ':') {
    // Non-capturing group.
    ++current_;
  } else if (*current_ == '=' || *current_ == '!') {
    // Positive or negative lookahead.
    RaiseError("Lookahead is not supported yet.");
  } else if (*current_ == '<' && current_ + 1 != end_ &&
             (current_[1] == '=' || current_[1] == '!')) {
    // Positive or negative lookbehind.
    RaiseError("Lookbehind is not supported yet.");
  } else if (*current_ == '<') {
    ++current_;
    while (current_ != end_ && isalpha(*current_)) {
      ++current_;
    }
    if (current_ == end_ || *current_ != '>') {
      RaiseError("Invalid named capturing group.");
    }
    // Just ignore the named of the group.
    ++current_;
  } else {
    // Group modifier flag.
    RaiseError("Group modifier flag is not supported yet.");
  }
}

std::string RegexConverter::Convert() {
  start_ = regex_codepoints_.data();
  current_ = start_;
  end_ = start_ + regex_codepoints_.size() - 1;
  bool is_empty = true;
  while (current_ != end_) {
    if (*current_ == '^') {
      if (current_ != start_) {
        RaiseWarning(
            "'^' should be at the start of the regex, but found in the middle. It is ignored."
        );
      }
      ++current_;
    } else if (*current_ == '$') {
      if (current_ != end_ - 1) {
        RaiseWarning(
            "'$' should be at the end of the regex, but found in the middle. It is ignored."
        );
      }
      ++current_;
    } else if (*current_ == '[') {
      is_empty = false;
      AddEBNFSegment(HandleCharacterClass());
    } else if (*current_ == '(') {
      is_empty = false;
      ++current_;
      ++parenthesis_level_;
      AddEBNFSegment("(");
      if (current_ != end_ && *current_ == '?') {
        ++current_;
        HandleGroupModifier();
      }
    } else if (*current_ == ')') {
      is_empty = false;
      if (parenthesis_level_ == 0) {
        RaiseError("Unmatched ')'");
      }
      // Empty alternative before ')' (e.g. "(a|)" or "(a|$)"): emit "" so it isn't a bare '|'.
      if (!result_ebnf_.empty() && result_ebnf_.back() == '|') {
        AddEBNFSegment("\"\"");
      }
      --parenthesis_level_;
      AddEBNFSegment(")");
      ++current_;
    } else if (*current_ == '*' || *current_ == '+' || *current_ == '?') {
      is_empty = false;
      result_ebnf_ += static_cast<char>(*current_);
      ++current_;
      if (current_ != end_ && *current_ == '?') {
        // Ignore the non-greedy modifier because our grammar handles all repetition numbers
        // non-deterministically.
        ++current_;
      }
      if (current_ != end_ &&
          (*current_ == '{' || *current_ == '*' || *current_ == '+' || *current_ == '?')) {
        RaiseError("Two consecutive repetition modifiers are not allowed.");
      }
    } else if (*current_ == '{') {
      is_empty = false;
      result_ebnf_ += HandleRepetitionRange();
      if (current_ != end_ && *current_ == '?') {
        // Still ignore the non-greedy modifier.
        ++current_;
      }
      if (current_ != end_ &&
          (*current_ == '{' || *current_ == '*' || *current_ == '+' || *current_ == '?')) {
        RaiseError("Two consecutive repetition modifiers are not allowed.");
      }
    } else if (*current_ == '|') {
      is_empty = false;
      // Empty alternative before '|': emit "" so there's no bare '|' on the left.
      // Covers leading ("^$|abc"), consecutive ("a||b") and group-start ("(|a)") cases.
      if (result_ebnf_.empty() || result_ebnf_.back() == '|' || result_ebnf_.back() == '(') {
        AddEBNFSegment("\"\"");
      }
      AddEBNFSegment("|");
      ++current_;
    } else if (*current_ == '\\') {
      is_empty = false;
      AddEBNFSegment(HandleEscape());
    } else if (*current_ == '.') {
      is_empty = false;
      AddEBNFSegment(R"([\u0000-\U0010FFFF])");
      ++current_;
    } else {
      is_empty = false;
      // Non-special characters are matched literally.
      AddEBNFSegment("\"" + EscapeString(*current_) + "\"");
      ++current_;
    }
  }
  if (parenthesis_level_ != 0) {
    RaiseError("The parenthesis is not closed.");
  }
  // Trailing empty alternative, e.g. "abc|": emit "" so it doesn't end with a bare '|'.
  if (!result_ebnf_.empty() && result_ebnf_.back() == '|') {
    AddEBNFSegment("\"\"");
  }
  if (is_empty) {
    AddEBNFSegment("\"\"");
  }
  return result_ebnf_;
}

namespace {
// Normalize ECMAScript character classes for product regex and JSON Schema patterns.
// Reject escapes that the lower-level converter would only warn about.
std::string normalize_characters(const std::string& pattern) {
  const std::string space =
      R"(\u0009-\u000d\u0020\u00a0\u1680\u2000-\u200a\u2028-\u2029\u202f\u205f\u3000\ufeff)";
  const std::string nonspace =
      R"(\u0000-\u0008\u000e-\u001f\u0021-\u009f\u00a1-\u167f\u1681-\u1fff\u200b-\u2027\u202a-\u202e\u2030-\u205e\u2060-\u2fff\u3001-\ufefe\uff00-\u{10ffff})";
  std::string result;
  bool in_class = false;
  for (size_t i = 0; i < pattern.size(); ++i) {
    char c = pattern[i];
    if (c == '\\') {
      if (++i == pattern.size()) throw std::invalid_argument("unfinished pattern escape");
      c = pattern[i];
      if (c == 's' || c == 'S') {
        result += (in_class ? "" : "[") + (c == 's' ? space : nonspace) + (in_class ? "" : "]");
      } else if (in_class && (c == '^' || c == '[' || c == '-' || c == 'b')) {
        result += c == '^' ? R"(\x5e)" : c == '[' ? R"(\x5b)" : c == '-' ? R"(\x2d)" : R"(\x08)";
      } else {
        if (std::string_view("dDwWfnrtvuxc^$.*+?\\()[]{}|/-").find(c) == std::string_view::npos)
          throw std::invalid_argument("unsupported pattern escape: \\" + std::string(1, c));
        if (c == 'x' || c == 'u') {
          const size_t digits = c == 'x' ? 2 : 4;
          if (i + digits >= pattern.size())
            throw std::invalid_argument("unfinished pattern escape");
          for (size_t j = 1; j <= digits; ++j)
            if (HexCharToInt(pattern[i + j]) < 0)
              throw std::invalid_argument("invalid hexadecimal pattern escape");
        }
        if (c == 'c' &&
            (i + 1 == pattern.size() || !((pattern[i + 1] >= 'A' && pattern[i + 1] <= 'Z') ||
                                          (pattern[i + 1] >= 'a' && pattern[i + 1] <= 'z'))))
          throw std::invalid_argument("invalid control character pattern escape");
        if (c == 'u' && i + 4 < pattern.size() &&
            (pattern[i + 1] == 'd' || pattern[i + 1] == 'D') && HexCharToInt(pattern[i + 2]) >= 8)
          throw std::invalid_argument(
              "surrogate escapes in patterns are not supported; use the Unicode character");
        result += '\\';
        result += c;
      }
    } else if (c == '\0') {
      result += R"(\x00)";
    } else if (c == '.' && !in_class) {
      result += R"([^\n\r\u2028\u2029])";
    } else {
      result += c;
      if (c == '[')
        in_class = true;
      else if (c == ']')
        in_class = false;
    }
  }
  return result;
}

// Anchors are supported at the ends of each top-level alternative. Other zero-width assertions
// require a different regex execution model and are rejected rather than erased by conversion.
std::string normalize_anchors(const std::string& pattern, bool full_match) {
  std::vector<std::string> branches;
  bool escaped = false, in_class = false;
  int depth = 0;
  std::size_t begin = 0;
  for (std::size_t i = 0; i <= pattern.size(); ++i) {
    if (i == pattern.size() || (!escaped && !in_class && depth == 0 && pattern[i] == '|')) {
      branches.push_back(pattern.substr(begin, i - begin));
      begin = i + 1;
      continue;
    }
    const char c = pattern[i];
    if (escaped) {
      escaped = false;
      continue;
    }
    if (c == '\\') {
      escaped = true;
      continue;
    }
    if (c == '[' && !in_class)
      in_class = true;
    else if (c == ']' && in_class)
      in_class = false;
    else if (!in_class && c == '(')
      ++depth;
    else if (!in_class && c == ')')
      --depth;
  }
  std::string result;
  bool first = true;
  for (auto branch : branches) {
    bool start = !branch.empty() && branch.front() == '^';
    if (start) branch.erase(0, 1);
    std::size_t slashes = 0;
    if (branch.size() > 1)
      for (std::size_t i = branch.size() - 1; i > 0 && branch[i - 1] == '\\'; --i) ++slashes;
    bool end = !branch.empty() && branch.back() == '$' && slashes % 2 == 0;
    if (end) branch.pop_back();
    escaped = false;
    in_class = false;
    for (char c : branch) {
      if (escaped) {
        escaped = false;
        continue;
      }
      if (c == '\\') {
        escaped = true;
        continue;
      }
      if (c == '[' && !in_class)
        in_class = true;
      else if (c == ']' && in_class)
        in_class = false;
      else if (!in_class && (c == '^' || c == '$'))
        throw std::invalid_argument("pattern anchors must bound a top-level alternative");
    }
    if (!first) result += "|";
    first = false;
    const std::string any = R"([^\uD800-\uDFFF]*)";
    result += full_match
                  ? branch
                  : (start ? "" : any) + std::string("(?:") + branch + ")" + (end ? "" : any);
  }
  return result;
}
}  // namespace

std::string NormalizeRegexPattern(const std::string& pattern, bool full_match) {
  return normalize_anchors(normalize_characters(pattern), full_match);
}

std::string RegexToEBNF(const std::string& regex, bool with_rule_name) {
  RegexConverter converter(regex);
  if (with_rule_name) {
    return "root ::= " + converter.Convert() + "\n";
  } else {
    return converter.Convert();
  }
}

}  // namespace xgrammar
