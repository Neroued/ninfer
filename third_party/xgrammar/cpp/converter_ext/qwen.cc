/*!
 *  Copyright (c) 2024 by Contributors
 * \file xgrammar/converter_ext/qwen.cc
 * \brief Qwen XML parameter format.
 */
#include "../json_schema_converter_ext.h"

namespace xgrammar {
namespace converter_ext {

XMLWrapper GetQwenXMLWrapper() { return {"<parameter=", ">", "\n", "\n</parameter>\n"}; }

}  // namespace converter_ext
}  // namespace xgrammar
