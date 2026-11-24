/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/regex_converter.h
 * \brief Convert a regex string to EBNF grammar string.
 */

#ifndef XGRAMMAR_REGEX_CONVERTER_H_
#define XGRAMMAR_REGEX_CONVERTER_H_

#include <string>

namespace xgrammar {

/*!
 * \brief Convert a regex string to EBNF grammar string.
 */
std::string RegexToEBNF(const std::string& regex, bool with_rule_name = true);

// Shared ECMAScript character/escape semantics. Full matching accepts only the complete
// string; search matching adds unanchored prefixes/suffixes for JSON Schema patterns.
std::string NormalizeRegexPattern(const std::string& pattern, bool full_match);

}  // namespace xgrammar

#endif  // XGRAMMAR_REGEX_CONVERTER_H_
