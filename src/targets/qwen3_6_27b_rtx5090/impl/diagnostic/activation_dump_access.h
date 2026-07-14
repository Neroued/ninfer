#pragma once

#include <ninfer/targets/qwen3_6_27b_rtx5090/package.h>
#include "targets/qwen3_6_27b_rtx5090/impl/schedule/text_context.h"
#include "targets/qwen3_6_27b_rtx5090/impl/schedule/vision_context.h"

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {

// Target-private diagnostic attachment. The product Engine never exposes this seam; the dedicated
// activation-dump executable uses it to observe the exact Program schedule it executes.
class ActivationDumpAccess {
public:
    static void attach(Program& program, void* context, schedule::TextTapCallback text,
                       schedule::VisionTapCallback vision = nullptr);
    static void detach(Program& program) noexcept;
};

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail
