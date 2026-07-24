#pragma once

// Include this once from an exact target translation unit after defining
// NINFER_QWEN36_VARIANT and NINFER_QWEN36_RUNTIME_NS. The body is shared source; the selected
// Variant is compile-time data and the only target-dependent calls are its three closed leaves.

#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/program.h"
#include "targets/qwen3_6/impl/runtime/api_impl.h"

#include "targets/qwen3_6/impl/runtime/layouts_impl.h"
#include "targets/qwen3_6/impl/runtime/dflash_context_impl.h"
#include "targets/qwen3_6/impl/runtime/text_context_impl.h"
#include "targets/qwen3_6/impl/runtime/vision_context_impl.h"
#include "targets/qwen3_6/impl/runtime/text_prefill_impl.h"
#include "targets/qwen3_6/impl/runtime/speculative_impl.h"
#include "targets/qwen3_6/impl/runtime/graph_impl.h"
#include "targets/qwen3_6/impl/runtime/dflash_impl.h"
#include "targets/qwen3_6/impl/runtime/decode_impl.h"
#include "targets/qwen3_6/impl/runtime/mtp_impl.h"
#include "targets/qwen3_6/impl/runtime/request_plan_impl.h"
#include "targets/qwen3_6/impl/runtime/program_impl.h"
