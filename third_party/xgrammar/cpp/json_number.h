#ifndef XGRAMMAR_JSON_NUMBER_H_
#define XGRAMMAR_JSON_NUMBER_H_

#include <xgrammar/grammar.h>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace xgrammar {

// Decimal arithmetic for schema input and compilation. No binary rounding is involved in
// comparing bounds, intersecting ranges, or selecting decimal coefficients.
struct DecimalNumber {
  bool negative = false;
  std::string digits = "0";
  int exponent = 0;  // value = signed digits * 10^exponent

  static DecimalNumber Parse(std::string_view text);
  static DecimalNumber Exact(double value);
  static DecimalNumber Published(double value);
  std::string Text() const;
  DecimalNumber Negated() const;
  bool Zero() const { return digits == "0"; }
  int Compare(const DecimalNumber& other) const;
  // Positive decimal / 10^scale, rounded and clamped to [0, cap].
  std::uint64_t Coefficient(int scale, bool upward, bool exclusive, std::uint64_t cap) const;
};

struct NumberBound {
  DecimalNumber value;
  bool exclusive = false;
};

struct NumberRange {
  std::optional<NumberBound> lower, upper;
  void Lower(DecimalNumber value, bool exclusive = false);
  void Upper(DecimalNumber value, bool exclusive = false);
  bool Empty() const;
  bool Contains(const DecimalNumber& value) const;
  bool HasPublishableValue() const;
  std::optional<std::pair<std::int64_t, std::int64_t>> Integers() const;
};

// The numeric language preserves the requested decimal interval and its value after the
// product's JSON parse/serialize round trip. Floating spellings use <=17 significant digits.
Grammar BoundedNumberGrammar(const NumberRange& range);
bool NumberSpellingPreserved(std::string_view source, double value);

}  // namespace xgrammar
#endif
