#include "json_number.h"

#include "grammar_builder.h"
#include "grammar_functor.h"
#include "json_schema_converter.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace xgrammar {
namespace {
void Normalize(DecimalNumber& value) {
  const auto first = value.digits.find_first_not_of('0');
  if (first == std::string::npos) {
    value = {};
    return;
  }
  value.digits.erase(0, first);
  while (value.digits.size() > 1 && value.digits.back() == '0') {
    value.digits.pop_back();
    ++value.exponent;
  }
}

void Multiply(std::string& digits, unsigned by) {
  unsigned carry = 0;
  for (std::size_t i = digits.size(); i-- > 0;) {
    const unsigned value = (digits[i] - '0') * by + carry;
    digits[i] = static_cast<char>('0' + value % 10);
    carry = value / 10;
  }
  if (carry) digits.insert(0, std::to_string(carry));
}

// Adjacent nonnegative binary64 values, represented exactly as finite decimals.
DecimalNumber MidpointMagnitude(DecimalNumber a, DecimalNumber b) {
  const int exponent = std::min(a.exponent, b.exponent);
  a.digits.append(a.exponent - exponent, '0');
  b.digits.append(b.exponent - exponent, '0');
  const auto size = std::max(a.digits.size(), b.digits.size());
  a.digits.insert(0, size - a.digits.size(), '0');
  b.digits.insert(0, size - b.digits.size(), '0');
  unsigned carry = 0;
  for (std::size_t i = size; i-- > 0;) {
    const unsigned value = a.digits[i] - '0' + b.digits[i] - '0' + carry;
    a.digits[i] = static_cast<char>('0' + value % 10);
    carry = value / 10;
  }
  if (carry) a.digits.insert(a.digits.begin(), '1');
  Multiply(a.digits, 5);
  a.exponent = exponent - 1;
  Normalize(a);
  return a;
}

DecimalNumber Midpoint(double a, double b) {
  if (a < 0 && b > 0) throw std::logic_error("nonadjacent midpoint crosses zero");
  auto result =
      MidpointMagnitude(DecimalNumber::Exact(std::abs(a)), DecimalNumber::Exact(std::abs(b)));
  result.negative = a < 0 || b < 0;
  return result;
}

constexpr std::uint64_t Pow10(unsigned n) {
  std::uint64_t value = 1;
  while (n--) value *= 10;
  return value;
}

// Convert a signed decimal bound to the nearest permitted signed integer. Clamp before any
// cast or negation, including the asymmetric INT64_MIN magnitude.
std::int64_t IntegerEndpoint(const NumberBound& bound, bool lower) {
  if (bound.value.Zero() && bound.exclusive) return lower ? 1 : -1;
  constexpr auto max = std::numeric_limits<std::int64_t>::max();
  const auto cap = static_cast<std::uint64_t>(max) + (bound.value.negative ? 1 : 0);
  const bool up = lower != bound.value.negative;
  auto magnitude = bound.value;
  magnitude.negative = false;
  const auto rounded = magnitude.Coefficient(0, up, bound.exclusive, cap);
  if (!bound.value.negative) return static_cast<std::int64_t>(rounded);
  return rounded == static_cast<std::uint64_t>(max) + 1 ? std::numeric_limits<std::int64_t>::min()
                                                        : -static_cast<std::int64_t>(rounded);
}

double Nearest(const DecimalNumber& value) {
  const auto text = value.Text();
  double result = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  if (parsed.ec == std::errc::result_out_of_range) {
    auto magnitude = value;
    magnitude.negative = false;
    if (magnitude.Compare(DecimalNumber::Parse("1")) < 0) return value.negative ? -0.0 : 0.0;
    return value.negative ? -std::numeric_limits<double>::max()
                          : std::numeric_limits<double>::max();
  }
  if (parsed.ec != std::errc{}) throw std::logic_error("invalid decimal bound");
  return result;
}

// Find the extreme binary64 whose published decimal value satisfies one bound. The source
// endpoints are already checked for lossless JSON representation by the schema adapter.
std::optional<double> FloatingEndpoint(const std::optional<NumberBound>& bound, bool lower) {
  const double limit = std::numeric_limits<double>::max();
  if (!bound) return lower ? -limit : limit;
  const auto admits = [&](double value) {
    if (!std::isfinite(value)) return false;
    const int cmp = DecimalNumber::Published(value).Compare(bound->value);
    return lower ? cmp > 0 || (cmp == 0 && !bound->exclusive)
                 : cmp < 0 || (cmp == 0 && !bound->exclusive);
  };
  double value = Nearest(bound->value);
  const double direction = lower ? INFINITY : -INFINITY;
  while (!admits(value)) {
    value = std::nextafter(value, direction);
    if (!std::isfinite(value)) return {};
  }
  for (;;) {
    const double neighbor = std::nextafter(value, -direction);
    if (!admits(neighbor)) return value;
    value = neighbor;
  }
}

class NumberGrammar {
 public:
  Grammar Build(const NumberRange& requested) {
    if (requested.Empty())
      throw JSONSchemaCompileError(SchemaErrorType::kUnsatisfiableSchema,
                                   "numeric interval is empty", "");
    std::vector<int32_t> choices;
    if (const auto integers = requested.Integers()) {
      choices.push_back(builder_.AddRuleRef(SubGrammarAdder::Apply(
          &builder_, Grammar::FromRegex(GenerateRangeRegex(integers->first, integers->second)))));
    }
    const auto first = FloatingEndpoint(requested.lower, true);
    const auto last = FloatingEndpoint(requested.upper, false);
    if (first && last && *first <= *last) {
      NumberRange floating = requested;
      const double before = std::nextafter(*first, -INFINITY);
      const double after = std::nextafter(*last, INFINITY);
      floating.Lower(
          std::isfinite(before) ? Midpoint(before, *first) : DecimalNumber::Exact(*first),
          std::isfinite(before));
      floating.Upper(std::isfinite(after) ? Midpoint(*last, after) : DecimalNumber::Exact(*last),
                     std::isfinite(after));
      if (!floating.Empty()) {
        Positive(floating, false, choices);
        NumberRange negative;
        if (floating.upper)
          negative.Lower(floating.upper->value.Negated(), floating.upper->exclusive);
        if (floating.lower)
          negative.Upper(floating.lower->value.Negated(), floating.lower->exclusive);
        Positive(negative, true, choices);
      }
      if (requested.Contains({})) {
        const auto zero = builder_.AddSequence(
            {builder_.AddChoices({builder_.AddEmptyStr(), builder_.AddByteString("-")}),
             builder_.AddByteString("0"),
             builder_.AddChoices(
                 {builder_.AddEmptyStr(),
                  builder_.AddSequence({builder_.AddByteString("."),
                                        builder_.AddRepeatFromExpr(
                                            "zeros", builder_.AddByteString("0"), 1, 16)})})});
        choices.push_back(zero);
      }
    }
    if (choices.empty())
      throw JSONSchemaCompileError(SchemaErrorType::kUnsupportedSchema,
                                   "numeric interval has no publishable value", "");
    return GrammarNormalizer::Apply(
        builder_.Get(builder_.AddRuleWithHint("root", builder_.AddChoices(choices))));
  }

 private:
  int32_t Digits(std::uint64_t lo, std::uint64_t hi, unsigned width, int dot = -1) {
    if (width == 0) return builder_.AddEmptyStr();
    if (dot == 0) return builder_.AddSequence({builder_.AddByteString("."), Digits(lo, hi, width)});
    if (lo == 0 && hi == Pow10(width) - 1) {
      const auto digits = [&](unsigned count) {
        return builder_.AddRepeatFromExpr("digits", builder_.AddCharacterClass({{'0', '9'}}), count,
                                          count);
      };
      if (dot > 0 && dot < static_cast<int>(width))
        return builder_.AddSequence(
            {digits(dot), builder_.AddByteString("."), digits(width - dot)});
      return digits(width);
    }
    const auto factor = Pow10(width - 1);
    const auto a = lo / factor, b = hi / factor;
    const auto branch = [&](std::uint64_t digit, std::uint64_t low, std::uint64_t high) {
      return builder_.AddSequence(
          {builder_.AddByteString(std::to_string(digit)), Digits(low, high, width - 1, dot - 1)});
    };
    if (a == b) return branch(a, lo % factor, hi % factor);
    std::vector<int32_t> choices{branch(a, lo % factor, factor - 1)};
    if (a + 1 < b) {
      choices.push_back(builder_.AddSequence(
          {builder_.AddCharacterClass(
               {{static_cast<int32_t>('0' + a + 1), static_cast<int32_t>('0' + b - 1)}}),
           Digits(0, factor - 1, width - 1, dot - 1)}));
    }
    choices.push_back(branch(b, 0, hi % factor));
    return builder_.AddChoices(choices);
  }

  int32_t Exponent(int lo, int hi) {
    std::vector<int32_t> choices;
    choices.push_back(builder_.AddRuleRef(
        SubGrammarAdder::Apply(&builder_, Grammar::FromRegex(GenerateRangeRegex(lo, hi)))));
    // Accept the usual JSON serializers' e+07/e-07 spellings too.
    if (hi >= 0) {
      choices.push_back(builder_.AddSequence(
          {builder_.AddByteString("+"),
           builder_.AddRuleRef(SubGrammarAdder::Apply(
               &builder_, Grammar::FromRegex(GenerateRangeRegex(std::max(lo, 0), hi))))}));
    }
    for (int i = std::max(lo, -9); i <= std::min(hi, 9); ++i) {
      choices.push_back(builder_.AddByteString((i < 0 ? "-0" : "0") + std::to_string(std::abs(i))));
      if (i >= 0) choices.push_back(builder_.AddByteString("+0" + std::to_string(i)));
    }
    return builder_.AddSequence(
        {builder_.AddCharacterClass({{'e', 'e'}, {'E', 'E'}}), builder_.AddChoices(choices)});
  }

  std::pair<std::uint64_t, std::uint64_t> Coefficients(const NumberRange& range, int exponent,
                                                       unsigned width) const {
    auto lo = Pow10(width - 1), hi = Pow10(width) - 1;
    const auto cap = Pow10(width);
    const int scale = exponent - static_cast<int>(width) + 1;
    if (range.lower && !range.lower->value.negative)
      lo = std::max(lo, range.lower->value.Coefficient(scale, true, range.lower->exclusive, cap));
    if (range.upper) {
      if (range.upper->value.negative || range.upper->value.Zero()) return {1, 0};
      hi = std::min(hi, range.upper->value.Coefficient(scale, false, range.upper->exclusive, cap));
    }
    return {lo, hi};
  }

  void Positive(const NumberRange& range, bool negative, std::vector<int32_t>& out) {
    std::vector<int32_t> choices;
    int full_begin = 0, full_end = -1;
    const auto full_significand = [&] {
      return builder_.AddSequence(
          {builder_.AddCharacterClass({{'1', '9'}}),
           builder_.AddChoices(
               {builder_.AddEmptyStr(),
                builder_.AddSequence(
                    {builder_.AddByteString("."),
                     builder_.AddRepeatFromExpr(
                         "fraction", builder_.AddCharacterClass({{'0', '9'}}), 1, 16)})})});
    };
    const auto flush = [&] {
      if (full_end >= full_begin) {
        choices.push_back(
            builder_.AddSequence({full_significand(), Exponent(full_begin, full_end)}));
        full_end = full_begin - 1;
      }
    };
    for (int exponent = -324; exponent <= 308; ++exponent) {
      const auto [wide_lo, wide_hi] = Coefficients(range, exponent, 17);
      if (wide_lo > wide_hi) {
        flush();
        continue;
      }
      const bool full = wide_lo == Pow10(16) && wide_hi == Pow10(17) - 1;
      if (full) {
        if (full_end < full_begin) full_begin = exponent;
        full_end = exponent;
      } else {
        flush();
      }
      std::vector<int32_t> significands;
      for (unsigned width = 1; width <= 17; ++width) {
        const auto [lo, hi] = Coefficients(range, exponent, width);
        if (lo > hi) continue;
        if (!full) significands.push_back(Digits(lo, hi, width, 1));
        if (exponent >= -6 && exponent < 17 && static_cast<int>(width) > exponent + 1) {
          if (exponent < 0) {
            choices.push_back(builder_.AddSequence(
                {builder_.AddByteString("0." + std::string(-exponent - 1, '0')),
                 Digits(lo, hi, width)}));
          } else {
            choices.push_back(Digits(lo, hi, width, exponent + 1));
          }
        }
      }
      if (!full && !significands.empty())
        choices.push_back(builder_.AddSequence(
            {builder_.AddChoices(significands), Exponent(exponent, exponent)}));
    }
    flush();
    if (!choices.empty()) {
      auto values = builder_.AddChoices(choices);
      if (negative) values = builder_.AddSequence({builder_.AddByteString("-"), values});
      out.push_back(values);
    }
  }

  GrammarBuilder builder_;
};
}  // namespace

DecimalNumber DecimalNumber::Parse(std::string_view text) {
  DecimalNumber result;
  result.digits.clear();
  if (!text.empty() && text.front() == '-') {
    result.negative = true;
    text.remove_prefix(1);
  }
  bool fraction = false;
  std::size_t pos = 0;
  for (; pos < text.size() && text[pos] != 'e' && text[pos] != 'E'; ++pos) {
    if (text[pos] == '.') {
      fraction = true;
      continue;
    }
    if (text[pos] < '0' || text[pos] > '9') throw std::invalid_argument("invalid decimal number");
    result.digits.push_back(text[pos]);
    if (fraction) --result.exponent;
  }
  if (result.digits.empty()) throw std::invalid_argument("empty decimal number");
  if (result.digits.find_first_not_of('0') == std::string::npos) return {};
  if (pos != text.size()) {
    auto suffix = text.substr(pos + 1);
    if (!suffix.empty() && suffix.front() == '+') suffix.remove_prefix(1);
    int exponent = 0;
    const auto parsed = std::from_chars(suffix.data(), suffix.data() + suffix.size(), exponent);
    if (parsed.ec != std::errc{} || parsed.ptr != suffix.data() + suffix.size() ||
        exponent < -10000 || exponent > 10000 || result.exponent < -10000)
      throw std::invalid_argument("decimal exponent exceeds supported range");
    result.exponent += exponent;
  }
  Normalize(result);
  return result;
}

DecimalNumber DecimalNumber::Exact(double value) {
  if (!std::isfinite(value)) throw std::invalid_argument("number must be finite");
  if (value == 0) return {};
  const auto bits = std::bit_cast<std::uint64_t>(value);
  const int exponent = static_cast<int>((bits >> 52) & 0x7ff);
  const auto mantissa = (bits & ((1ULL << 52) - 1)) | (exponent ? 1ULL << 52 : 0);
  DecimalNumber result{value < 0, std::to_string(mantissa), 0};
  const int power = (exponent ? exponent - 1023 : -1022) - 52;
  for (int i = 0; i < std::abs(power); ++i) Multiply(result.digits, power < 0 ? 5 : 2);
  if (power < 0) result.exponent = power;
  Normalize(result);
  return result;
}

DecimalNumber DecimalNumber::Published(double value) {
  if (!std::isfinite(value)) throw std::invalid_argument("number must be finite");
  return Parse(nlohmann::ordered_json(value).dump());
}

std::string DecimalNumber::Text() const {
  return (negative ? "-" : "") + digits + "e" + std::to_string(exponent);
}

DecimalNumber DecimalNumber::Negated() const {
  auto result = *this;
  if (!Zero()) result.negative = !negative;
  return result;
}

int DecimalNumber::Compare(const DecimalNumber& other) const {
  if (Zero() || other.Zero()) {
    if (Zero() && other.Zero()) return 0;
    return Zero() ? (other.negative ? 1 : -1) : (negative ? -1 : 1);
  }
  if (negative != other.negative) return negative ? -1 : 1;
  const int sign = negative ? -1 : 1;
  const auto order = static_cast<long long>(digits.size()) + exponent;
  const auto other_order = static_cast<long long>(other.digits.size()) + other.exponent;
  if (order != other_order) return order < other_order ? -sign : sign;
  for (std::size_t i = 0; i < std::max(digits.size(), other.digits.size()); ++i) {
    const char a = i < digits.size() ? digits[i] : '0';
    const char b = i < other.digits.size() ? other.digits[i] : '0';
    if (a != b) return a < b ? -sign : sign;
  }
  return 0;
}

std::uint64_t DecimalNumber::Coefficient(int scale, bool upward, bool exclusive,
                                         std::uint64_t cap) const {
  if (negative) throw std::logic_error("negative decimal coefficient");
  if (Zero()) return upward && exclusive ? 1 : 0;
  const auto whole = static_cast<long long>(digits.size()) + exponent - scale;
  if (whole > 20) return cap;
  std::uint64_t value = 0;
  for (long long i = 0; i < whole; ++i) {
    const unsigned digit = i < static_cast<long long>(digits.size()) ? digits[i] - '0' : 0;
    if (value > cap / 10 || (value == cap / 10 && digit > cap % 10)) return cap;
    value = value * 10 + digit;
  }
  const bool remainder = whole < static_cast<long long>(digits.size());
  if (upward && (remainder || exclusive)) return value == cap ? cap : value + 1;
  if (!upward && exclusive && !remainder && value) return value - 1;
  return value;
}

void NumberRange::Lower(DecimalNumber value, bool exclusive) {
  const int cmp = lower ? value.Compare(lower->value) : 1;
  if (cmp > 0 || (cmp == 0 && exclusive)) lower = NumberBound{std::move(value), exclusive};
}

void NumberRange::Upper(DecimalNumber value, bool exclusive) {
  const int cmp = upper ? value.Compare(upper->value) : -1;
  if (cmp < 0 || (cmp == 0 && exclusive)) upper = NumberBound{std::move(value), exclusive};
}

bool NumberRange::Empty() const {
  if (!lower || !upper) return false;
  const int cmp = lower->value.Compare(upper->value);
  return cmp > 0 || (cmp == 0 && (lower->exclusive || upper->exclusive));
}

bool NumberRange::Contains(const DecimalNumber& value) const {
  if (lower) {
    const int cmp = value.Compare(lower->value);
    if (cmp < 0 || (cmp == 0 && lower->exclusive)) return false;
  }
  if (upper) {
    const int cmp = value.Compare(upper->value);
    if (cmp > 0 || (cmp == 0 && upper->exclusive)) return false;
  }
  return true;
}

std::optional<std::pair<std::int64_t, std::int64_t>> NumberRange::Integers() const {
  if (Empty()) return {};
  const auto lo = lower ? IntegerEndpoint(*lower, true) : std::numeric_limits<std::int64_t>::min();
  const auto hi = upper ? IntegerEndpoint(*upper, false) : std::numeric_limits<std::int64_t>::max();
  if (lo > hi || !Contains(DecimalNumber::Parse(std::to_string(lo))) ||
      !Contains(DecimalNumber::Parse(std::to_string(hi))))
    return {};
  return std::pair{lo, hi};
}

bool NumberRange::HasPublishableValue() const {
  if (Empty()) return false;
  if (Integers()) return true;
  const auto first = FloatingEndpoint(lower, true);
  const auto last = FloatingEndpoint(upper, false);
  return first && last && *first <= *last;
}

Grammar BoundedNumberGrammar(const NumberRange& range) { return NumberGrammar().Build(range); }

bool NumberSpellingPreserved(std::string_view source, double value) {
  // binary64 guarantees decimal -> binary -> decimal round trips for digits10 significant
  // digits in its normal range. Most request numbers need no serialization or allocation.
  unsigned digits = 0, significant = 0;
  for (char c : source) {
    if (c == 'e' || c == 'E') break;
    if (c < '0' || c > '9' || (digits == 0 && c == '0')) continue;
    ++digits;
    if (c != '0') significant = digits;
  }
  if (!significant) return value == 0;
  if (significant <= std::numeric_limits<double>::digits10 && std::isnormal(value)) return true;
  try {
    return DecimalNumber::Parse(source).Compare(DecimalNumber::Published(value)) == 0;
  } catch (const std::exception&) {
    return false;
  }
}
}  // namespace xgrammar
